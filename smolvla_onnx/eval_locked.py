#!/usr/bin/env python3
"""Locked evaluation harness for Track B Robotics (SmolVLA -> ONNX / TensorRT).

Evaluates exported ONNX subgraphs (vision_encoder, vlm_prefix, action_expert_step)
against cached deterministic PyTorch FP32 reference outputs from
`lerobot/smolvla_base` (pinned revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`).

DO NOT MODIFY THIS FILE ONCE LOCKED (chmod 444).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

MODEL_REPO_ID = "lerobot/smolvla_base"
MODEL_REVISION = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"

DEFAULT_SEEDS = [11, 23, 37, 41, 53]
DEFAULT_NUM_BATCHES = 8
DEFAULT_BATCH_SIZE = 1

NUM_CAMERAS = 3
IMAGE_SIZE = 512
LANG_MAX_LEN = 48
MAX_STATE_DIM = 32
MAX_ACTION_DIM = 32
CHUNK_SIZE = 50
NUM_EULER_STEPS = 10
NUM_VLM_LAYERS = 16
NUM_KV_HEADS = 5
HEAD_DIM = 64
PREFIX_LEN = 241  # 64 * 3 + 48 + 1

TRACK_DIR = Path(__file__).resolve().parent
REF_CACHE_PATH = TRACK_DIR / "datasets" / "held_out" / "smolvla_ref_batches.pt"
DEFAULT_BASELINE_JSON = TRACK_DIR / "baseline_pytorch.json"

CANONICAL_ONNX_FILES = {
    "vision": "smolvla_vision_encoder.onnx",
    "prefix": "smolvla_vlm_prefix.onnx",
    "expert": "smolvla_action_expert_step.onnx",
}

TASK_PROMPTS = [
    "pick up the red block and place it on the plate\n",
    "open the top drawer of the cabinet\n",
    "put the white mug on the wooden tray\n",
    "push the blue box to the left side of the table\n",
    "grasp the bowl and place it on top of the stove\n",
    "close the microwave door completely\n",
    "pick up the green cube and stack it on the yellow cube\n",
    "slide the lid onto the saucepan\n",
]

ACCEPTANCE_THRESHOLDS = {
    "fp32": {"actions_max_abs_err": 1e-4, "actions_cosine_sim": 0.99999},
    "fp16": {"actions_cosine_sim": 0.999},
    "int8": {"actions_cosine_sim": 0.995},
}

JETSON_ORIN_NANO_MEM_MB = 8192.0
JETSON_SAFE_BUDGET_MB = 6144.0


def _batch_seed(index: int, seeds: list[int]) -> int:
    if index < len(seeds):
        return seeds[index]
    return seeds[index % len(seeds)] + 100 * (index // len(seeds))


def _query_process_gpu_mem_mb(pid: int) -> float | None:
    """Query process GPU memory usage in MiB via nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
    except Exception:
        return None
    total_mb = 0.0
    found = False
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            row_pid = int(parts[0])
            mem_val = float(parts[1])
        except ValueError:
            continue
        if row_pid == pid:
            total_mb += mem_val
            found = True
    return total_mb if found else 0.0


class ResourceMonitor:
    """Samples process RSS (and optional GPU process VRAM) in a background thread."""

    def __init__(self, track_gpu: bool = True, interval_s: float = 0.02) -> None:
        self.pid = os.getpid()
        self.proc = psutil.Process(self.pid)
        self.track_gpu = track_gpu
        self.interval_s = interval_s
        self.rss_start_mb = self._read_rss_mb()
        self.rss_peak_mb = self.rss_start_mb
        self.gpu_start_mb = _query_process_gpu_mem_mb(self.pid) if track_gpu else None
        self.gpu_peak_mb = self.gpu_start_mb
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _read_rss_mb(self) -> float:
        return self.proc.memory_info().rss / (1024.0**2)

    def _poll_loop(self) -> None:
        tick = 0
        while not self._stop_event.wait(self.interval_s):
            rss = self._read_rss_mb()
            if rss > self.rss_peak_mb:
                self.rss_peak_mb = rss
            tick += 1
            if self.track_gpu and tick % 5 == 0:
                gpu = _query_process_gpu_mem_mb(self.pid)
                if gpu is not None and (self.gpu_peak_mb is None or gpu > self.gpu_peak_mb):
                    self.gpu_peak_mb = gpu

    def __enter__(self) -> "ResourceMonitor":
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        rss_end = self._read_rss_mb()
        if rss_end > self.rss_peak_mb:
            self.rss_peak_mb = rss_end
        if self.track_gpu:
            gpu_end = _query_process_gpu_mem_mb(self.pid)
            if gpu_end is not None and (self.gpu_peak_mb is None or gpu_end > self.gpu_peak_mb):
                self.gpu_peak_mb = gpu_end

    def summary(self) -> dict[str, float | None]:
        ru_maxrss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        gpu_delta = None
        if self.gpu_peak_mb is not None and self.gpu_start_mb is not None:
            gpu_delta = max(0.0, self.gpu_peak_mb - self.gpu_start_mb)
        return {
            "rss_start_mb": round(self.rss_start_mb, 2),
            "rss_peak_mb": round(self.rss_peak_mb, 2),
            "rss_delta_mb": round(max(0.0, self.rss_peak_mb - self.rss_start_mb), 2),
            "process_ru_maxrss_mb": round(ru_maxrss_mb, 2),
            "nvml_gpu_start_mb": round(self.gpu_start_mb, 2) if self.gpu_start_mb is not None else None,
            "nvml_gpu_peak_mb": round(self.gpu_peak_mb, 2) if self.gpu_peak_mb is not None else None,
            "nvml_gpu_delta_mb": round(gpu_delta, 2) if gpu_delta is not None else None,
        }


@dataclass
class AccuracyMetrics:
    max_abs_err: float
    rel_max_abs_err: float
    mae: float
    rmse: float
    cosine_sim: float


def compute_accuracy_metrics(pred: np.ndarray, ref: np.ndarray) -> AccuracyMetrics:
    """Compute MaxAbsErr, relative MaxAbsErr, MAE, RMSE, and mean per-sample cosine similarity."""
    pred_f64 = np.asarray(pred, dtype=np.float64)
    ref_f64 = np.asarray(ref, dtype=np.float64)
    if pred_f64.shape != ref_f64.shape:
        raise ValueError(f"Shape mismatch: pred {pred_f64.shape} vs ref {ref_f64.shape}")

    diff = np.abs(pred_f64 - ref_f64)
    max_abs_err = float(np.max(diff))
    ref_max = float(np.max(np.abs(ref_f64)))
    rel_max_abs_err = float(max_abs_err / max(ref_max, 1e-12))
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff**2)))

    bsize = pred_f64.shape[0]
    p_flat = pred_f64.reshape(bsize, -1)
    r_flat = ref_f64.reshape(bsize, -1)
    p_norm = np.linalg.norm(p_flat, axis=1)
    r_norm = np.linalg.norm(r_flat, axis=1)
    denom = np.maximum(p_norm * r_norm, 1e-12)
    cos_per_sample = np.sum(p_flat * r_flat, axis=1) / denom
    cosine_sim = float(np.mean(cos_per_sample))

    return AccuracyMetrics(
        max_abs_err=max_abs_err,
        rel_max_abs_err=rel_max_abs_err,
        mae=mae,
        rmse=rmse,
        cosine_sim=cosine_sim,
    )


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": round(float(np.mean(arr)), 3),
        "std_ms": round(float(np.std(arr)), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
    }


def _resolve_model_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(MODEL_REPO_ID, revision=MODEL_REVISION, local_files_only=True)
    except Exception:
        local_dir = snapshot_download(MODEL_REPO_ID, revision=MODEL_REVISION, local_files_only=False)
    return Path(local_dir)


def _load_smolvla_policy(device: str, dtype: torch.dtype | None):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    local_dir = _resolve_model_snapshot()
    policy = SmolVLAPolicy.from_pretrained(str(local_dir), local_files_only=True)
    policy.eval()
    if dtype is not None:
        policy = policy.to(device=device, dtype=dtype)
    else:
        policy = policy.to(device=device)
    return policy


def _build_synthetic_camera_tensor(
    generator: torch.Generator, batch_size: int, num_cameras: int = NUM_CAMERAS
) -> torch.Tensor:
    """Create deterministic smooth-pattern + texture RGB images in [-1.0, 1.0]."""
    grid_y = torch.linspace(-1.0, 1.0, IMAGE_SIZE, dtype=torch.float32)[None, None, :, None]
    grid_x = torch.linspace(-1.0, 1.0, IMAGE_SIZE, dtype=torch.float32)[None, None, None, :]
    freq = torch.rand((batch_size * num_cameras, 3, 1, 1), generator=generator, dtype=torch.float32) * 3.0 + 1.0
    phase = torch.rand((batch_size * num_cameras, 3, 1, 1), generator=generator, dtype=torch.float32) * (2.0 * math.pi)
    pattern = 0.6 * torch.sin(freq * (grid_x + grid_y) + phase)
    noise = (torch.rand((batch_size * num_cameras, 3, IMAGE_SIZE, IMAGE_SIZE), generator=generator, dtype=torch.float32) * 2.0 - 1.0) * 0.4
    return torch.clamp(pattern + noise, -1.0, 1.0)


def generate_and_cache_reference_batches(
    cache_path: Path,
    num_batches: int,
    batch_size: int,
    seeds: list[int],
) -> dict[str, Any]:
    """Generate deterministic observation batches and PyTorch FP32 reference outputs."""
    from lerobot.policies.common.vla_utils import clone_past_key_values, make_att_2d_masks

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        policy = _load_smolvla_policy(device=device, dtype=torch.float32)
        tokenizer = policy.model.vlm_with_expert.processor.tokenizer

        batches: list[dict[str, torch.Tensor]] = []
        with torch.no_grad():
            for idx in range(num_batches):
                seed = _batch_seed(idx, seeds)
                gen = torch.Generator(device="cpu").manual_seed(seed)

                pixel_values = _build_synthetic_camera_tensor(gen, batch_size, NUM_CAMERAS).to(device)
                images = [
                    pixel_values.view(batch_size, NUM_CAMERAS, 3, IMAGE_SIZE, IMAGE_SIZE)[:, cam_i]
                    for cam_i in range(NUM_CAMERAS)
                ]
                img_masks_tensor = torch.ones((batch_size, NUM_CAMERAS), dtype=torch.bool, device=device)
                img_masks = [img_masks_tensor[:, cam_i] for cam_i in range(NUM_CAMERAS)]

                prompt = TASK_PROMPTS[idx % len(TASK_PROMPTS)]
                enc = tokenizer(
                    [prompt] * batch_size,
                    max_length=LANG_MAX_LEN,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                lang_tokens = enc["input_ids"].to(device=device, dtype=torch.int64)
                lang_masks = enc["attention_mask"].to(device=device, dtype=torch.bool)

                state = torch.zeros((batch_size, MAX_STATE_DIM), dtype=torch.float32, device=device)
                state[:, :6] = torch.randn((batch_size, 6), generator=gen, dtype=torch.float32).to(device) * 0.5
                x_1 = torch.randn((batch_size, CHUNK_SIZE, MAX_ACTION_DIM), generator=gen, dtype=torch.float32).to(device)

                # (a) Vision encoder embeddings [B*3, 64, 960]
                raw_img_emb = policy.model.vlm_with_expert.embed_image(pixel_values)
                ref_image_embeddings = raw_img_emb * math.sqrt(raw_img_emb.shape[-1])

                # (b) Prefix KV cache [16, B, 5, 241, 64]
                img_embs_seq = ref_image_embeddings.reshape(batch_size, NUM_CAMERAS * 64, 960)
                lang_emb = policy.model.vlm_with_expert.embed_language_tokens(lang_tokens)
                lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
                state_emb = policy.model.state_proj(state)[:, None, :]
                prefix_embs = torch.cat([img_embs_seq, lang_emb, state_emb], dim=1)

                img_pad_masks = torch.cat(
                    [img_masks[c][:, None].expand(batch_size, 64) for c in range(NUM_CAMERAS)], dim=1
                )
                state_pad_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
                prefix_pad_masks = torch.cat([img_pad_masks, lang_masks, state_pad_mask], dim=1)

                att_masks = torch.zeros((batch_size, PREFIX_LEN), dtype=torch.bool, device=device)
                att_masks[:, -1] = True
                prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, att_masks)
                prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

                _, pkv = policy.model.vlm_with_expert.forward(
                    attention_mask=prefix_att_2d_masks,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=True,
                )
                ref_past_keys = torch.stack([layer.keys for layer in pkv.layers], dim=0)
                ref_past_values = torch.stack([layer.values for layer in pkv.layers], dim=0)

                # (c) Single-step flow-matching velocity at t=1.0, x_t=x_1
                pkv_step = clone_past_key_values(pkv)
                t0 = torch.ones((batch_size,), dtype=torch.float32, device=device)
                ref_v_t = policy.model.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=pkv_step,
                    x_t=x_1,
                    timestep=t0,
                )

                # (d) Full 10-step Euler denoised action chunk
                pkv_euler = clone_past_key_values(pkv)
                x_t = x_1.clone()
                dt = -1.0 / NUM_EULER_STEPS
                for step in range(NUM_EULER_STEPS):
                    t_cur = torch.full((batch_size,), 1.0 + step * dt, dtype=torch.float32, device=device)
                    v_cur = policy.model.denoise_step(
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=pkv_euler,
                        x_t=x_t,
                        timestep=t_cur,
                    )
                    x_t = x_t + dt * v_cur
                ref_actions = x_t

                # Consistency check against full policy.model.sample_actions
                e2e_actions = policy.model.sample_actions(
                    images, img_masks, lang_tokens, lang_masks, state, noise=x_1
                )
                consistency_err = float((ref_actions - e2e_actions).abs().max().item())
                if consistency_err >= 5e-3:
                    raise RuntimeError(
                        f"Reference decomposed pipeline diverged from sample_actions: {consistency_err}"
                    )

                batches.append(
                    {
                        "seed": seed,
                        "prompt": prompt,
                        "pixel_values": pixel_values.cpu().contiguous(),
                        "img_masks": img_masks_tensor.cpu().contiguous(),
                        "lang_tokens": lang_tokens.cpu().contiguous(),
                        "lang_masks": lang_masks.cpu().contiguous(),
                        "state": state.cpu().contiguous(),
                        "x_1": x_1.cpu().contiguous(),
                        "ref_image_embeddings": ref_image_embeddings.cpu().contiguous(),
                        "ref_past_keys": ref_past_keys.cpu().contiguous(),
                        "ref_past_values": ref_past_values.cpu().contiguous(),
                        "ref_prefix_pad_masks": prefix_pad_masks.cpu().contiguous(),
                        "ref_v_t": ref_v_t.cpu().contiguous(),
                        "ref_actions": ref_actions.cpu().contiguous(),
                        "ref_e2e_consistency_err": consistency_err,
                    }
                )

        payload = {
            "metadata": {
                "repo_id": MODEL_REPO_ID,
                "revision": MODEL_REVISION,
                "seeds": seeds,
                "num_batches": num_batches,
                "batch_size": batch_size,
                "generated_on": device,
                "shapes": {
                    "pixel_values": [batch_size * NUM_CAMERAS, 3, IMAGE_SIZE, IMAGE_SIZE],
                    "image_embeddings": [batch_size * NUM_CAMERAS, 64, 960],
                    "past_keys": [NUM_VLM_LAYERS, batch_size, NUM_KV_HEADS, PREFIX_LEN, HEAD_DIM],
                    "past_values": [NUM_VLM_LAYERS, batch_size, NUM_KV_HEADS, PREFIX_LEN, HEAD_DIM],
                    "v_t": [batch_size, CHUNK_SIZE, MAX_ACTION_DIM],
                    "actions": [batch_size, CHUNK_SIZE, MAX_ACTION_DIM],
                },
            },
            "batches": batches,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_path)
        return payload
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_or_create_reference_batches(
    cache_path: Path,
    num_batches: int,
    batch_size: int,
    seeds: list[int],
) -> dict[str, Any]:
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        meta = payload.get("metadata", {})
        if (
            meta.get("repo_id") == MODEL_REPO_ID
            and meta.get("revision") == MODEL_REVISION
            and meta.get("batch_size") == batch_size
            and len(payload.get("batches", [])) >= num_batches
        ):
            return {
                "metadata": meta,
                "batches": payload["batches"][:num_batches],
            }
        raise ValueError(
            f"Cached reference file {cache_path} has incompatible metadata {meta}; "
            f"requested num_batches={num_batches}, batch_size={batch_size}."
        )
    return generate_and_cache_reference_batches(cache_path, num_batches, batch_size, seeds)


def _sync_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_pytorch_policy(
    ref_batches: list[dict[str, Any]],
    mode: str,
    device: str,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    """Benchmark PyTorch SmolVLA (`fp32` or `bf16` native) with per-subgraph timing & accuracy."""
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for PyTorch benchmark, but torch.cuda.is_available() is False.")

    dtype = torch.float32 if mode == "fp32" else None  # None preserves official BF16 backbone + F32 heads
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with ResourceMonitor(track_gpu=(device == "cuda")) as monitor:
        policy = _load_smolvla_policy(device=device, dtype=dtype)
        rss_after_load_mb = psutil.Process(os.getpid()).memory_info().rss / (1024.0**2)

        def _prepare_inputs(b: dict[str, Any]) -> tuple[Any, ...]:
            pv = b["pixel_values"].to(device=device)
            bsize = b["state"].shape[0]
            imgs = [pv.view(bsize, NUM_CAMERAS, 3, IMAGE_SIZE, IMAGE_SIZE)[:, c] for c in range(NUM_CAMERAS)]
            img_masks_t = b["img_masks"].to(device=device)
            img_masks = [img_masks_t[:, c] for c in range(NUM_CAMERAS)]
            lang_tokens = b["lang_tokens"].to(device=device)
            lang_masks = b["lang_masks"].to(device=device)
            state = b["state"].to(device=device)
            x_1 = b["x_1"].to(device=device)
            return pv, imgs, img_masks, lang_tokens, lang_masks, state, x_1

        def _run_decomposed_once(inputs: tuple[Any, ...]) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
            pv, imgs, img_masks, lang_tokens, lang_masks, state, x_1 = inputs
            bsize = state.shape[0]
            _sync_device(device)
            t0 = time.perf_counter()

            # 1. Vision encoder
            raw_emb = policy.model.vlm_with_expert.embed_image(pv)
            img_embs_scaled = raw_emb * math.sqrt(raw_emb.shape[-1])
            _sync_device(device)
            t1 = time.perf_counter()

            # 2. VLM prefix KV prefill
            img_seq = img_embs_scaled.reshape(bsize, NUM_CAMERAS * 64, 960)
            lang_emb = policy.model.vlm_with_expert.embed_language_tokens(lang_tokens)
            lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
            state_emb = policy.model.state_proj(state)[:, None, :]
            prefix_embs = torch.cat([img_seq, lang_emb, state_emb], dim=1)
            img_pad = torch.cat([img_masks[c][:, None].expand(bsize, 64) for c in range(NUM_CAMERAS)], dim=1)
            state_pad = torch.ones((bsize, 1), dtype=torch.bool, device=device)
            prefix_pad_masks = torch.cat([img_pad, lang_masks, state_pad], dim=1)
            att_masks = torch.zeros((bsize, PREFIX_LEN), dtype=torch.bool, device=device)
            att_masks[:, -1] = True
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, att_masks)
            prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
            _, pkv = policy.model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d,
                position_ids=prefix_pos,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            _sync_device(device)
            t2 = time.perf_counter()

            # 3. 10-step Euler denoising
            x_t = x_1.clone()
            dt = -1.0 / NUM_EULER_STEPS
            v_0 = None
            for step in range(NUM_EULER_STEPS):
                t_val = torch.full((bsize,), 1.0 + step * dt, dtype=torch.float32, device=device)
                v_t = policy.model.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=pkv,
                    x_t=x_t,
                    timestep=t_val,
                )
                if step == 0:
                    v_0 = v_t.clone()
                x_t = x_t + dt * v_t
            _sync_device(device)
            t3 = time.perf_counter()

            timings = {
                "vision_ms": (t1 - t0) * 1000.0,
                "prefix_ms": (t2 - t1) * 1000.0,
                "expert_10steps_ms": (t3 - t2) * 1000.0,
                "total_ms": (t3 - t0) * 1000.0,
            }
            outputs = {
                "image_embeddings": img_embs_scaled,
                "past_keys": torch.stack([l.keys for l in pkv.layers], dim=0),
                "past_values": torch.stack([l.values for l in pkv.layers], dim=0),
                "v_t": v_0,
                "actions": x_t,
            }
            return outputs, timings

        device_inputs = [_prepare_inputs(b) for b in ref_batches]

        # Accuracy pass over all ref batches
        acc_lists: dict[str, list[AccuracyMetrics]] = {
            "vision_image_embeddings": [],
            "prefix_past_keys": [],
            "prefix_past_values": [],
            "expert_single_step_v_t": [],
            "actions_10step": [],
        }
        with torch.no_grad():
            for b, inp in zip(ref_batches, device_inputs, strict=True):
                outs, _ = _run_decomposed_once(inp)
                acc_lists["vision_image_embeddings"].append(
                    compute_accuracy_metrics(outs["image_embeddings"].float().cpu().numpy(), b["ref_image_embeddings"].numpy())
                )
                acc_lists["prefix_past_keys"].append(
                    compute_accuracy_metrics(outs["past_keys"].float().cpu().numpy(), b["ref_past_keys"].numpy())
                )
                acc_lists["prefix_past_values"].append(
                    compute_accuracy_metrics(outs["past_values"].float().cpu().numpy(), b["ref_past_values"].numpy())
                )
                acc_lists["expert_single_step_v_t"].append(
                    compute_accuracy_metrics(outs["v_t"].float().cpu().numpy(), b["ref_v_t"].numpy())
                )
                acc_lists["actions_10step"].append(
                    compute_accuracy_metrics(outs["actions"].float().cpu().numpy(), b["ref_actions"].numpy())
                )

            # Warmup
            for w in range(warmup):
                _run_decomposed_once(device_inputs[w % len(device_inputs)])

            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            # Timed runs
            vision_times, prefix_times, expert_times, total_times = [], [], [], []
            for r in range(runs):
                _, tm = _run_decomposed_once(device_inputs[r % len(device_inputs)])
                vision_times.append(tm["vision_ms"])
                prefix_times.append(tm["prefix_ms"])
                expert_times.append(tm["expert_10steps_ms"])
                total_times.append(tm["total_ms"])

        peak_vram_alloc_mb = (
            torch.cuda.max_memory_allocated() / (1024.0**2) if device == "cuda" else 0.0
        )
        peak_vram_res_mb = (
            torch.cuda.max_memory_reserved() / (1024.0**2) if device == "cuda" else 0.0
        )
        del policy, device_inputs
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    mem_stats = monitor.summary()
    mem_stats["rss_after_load_mb"] = round(rss_after_load_mb, 2)
    mem_stats["torch_peak_vram_allocated_mb"] = round(peak_vram_alloc_mb, 2)
    mem_stats["torch_peak_vram_reserved_mb"] = round(peak_vram_res_mb, 2)

    lat_summary = summarize_latencies(total_times)
    mean_ms = lat_summary["mean_ms"]
    chunks_per_sec = round(1000.0 / max(mean_ms, 1e-6), 3)
    actions_per_sec = round(chunks_per_sec * CHUNK_SIZE, 3)

    acc_summary = {}
    for key, items in acc_lists.items():
        acc_summary[key] = {
            "max_abs_err": max(m.max_abs_err for m in items),
            "rel_max_abs_err": max(m.rel_max_abs_err for m in items),
            "mae": float(np.mean([m.mae for m in items])),
            "rmse": float(np.mean([m.rmse for m in items])),
            "cosine_sim": float(np.mean([m.cosine_sim for m in items])),
            "min_cosine_sim": min(m.cosine_sim for m in items),
        }

    return {
        "backend": f"pytorch_{mode}_{device}",
        "mode": mode,
        "device": device,
        "warmup_runs": warmup,
        "timed_runs": runs,
        "latency": lat_summary,
        "breakdown_ms": {
            "vision_mean_ms": round(float(np.mean(vision_times)), 3),
            "prefix_mean_ms": round(float(np.mean(prefix_times)), 3),
            "expert_10steps_mean_ms": round(float(np.mean(expert_times)), 3),
            "expert_per_step_mean_ms": round(float(np.mean(expert_times)) / NUM_EULER_STEPS, 3),
        },
        "throughput": {
            "chunks_per_sec": chunks_per_sec,
            "control_actions_per_sec": actions_per_sec,
        },
        "memory": mem_stats,
        "accuracy_vs_fp32_ref": acc_summary,
    }


def _resolve_onnx_files(onnx_dir: Path) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for key, fname in CANONICAL_ONNX_FILES.items():
        p = onnx_dir / fname
        if p.exists():
            resolved[key] = p
            continue
        stem = fname.removesuffix(".onnx")
        candidates = sorted(onnx_dir.glob(f"{stem}*.onnx"))
        if not candidates:
            raise FileNotFoundError(
                f"Missing ONNX subgraph '{fname}' (or '{stem}*.onnx') in {onnx_dir}"
            )
        resolved[key] = candidates[0]
    return resolved


def _ort_type_to_numpy(ort_type: str) -> np.dtype:
    if "float16" in ort_type:
        return np.float16
    if "float" in ort_type:
        return np.float32
    if "int64" in ort_type:
        return np.int64
    if "int32" in ort_type:
        return np.int32
    if "bool" in ort_type:
        return np.bool_
    raise ValueError(f"Unsupported ORT tensor type: {ort_type}")


def _ort_type_to_torch(ort_type: str) -> torch.dtype:
    if "float16" in ort_type:
        return torch.float16
    if "float" in ort_type:
        return torch.float32
    if "int64" in ort_type:
        return torch.int64
    if "int32" in ort_type:
        return torch.int32
    if "bool" in ort_type:
        return torch.bool
    raise ValueError(f"Unsupported ORT tensor type: {ort_type}")


def _make_ort_providers(device: str, provider_choice: str) -> list[Any]:
    import onnxruntime as ort

    if device == "cuda" and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    available = ort.get_available_providers()
    if device == "cpu" or provider_choice == "cpu":
        return ["CPUExecutionProvider"]
    if provider_choice == "tensorrt":
        if "TensorrtExecutionProvider" not in available:
            raise RuntimeError(f"TensorrtExecutionProvider unavailable; available={available}")
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(f"CUDAExecutionProvider unavailable; available={available}")
    return [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]


def _dir_onnx_size_mb(onnx_dir: Path) -> float:
    total = 0
    for f in onnx_dir.iterdir():
        if f.is_file() and (f.suffix in {".onnx", ".data"} or ".onnx" in f.name):
            total += f.stat().st_size
    return round(total / (1024.0**2), 2)


def _cast_feed(session: Any, raw_feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    feed = {}
    for inp in session.get_inputs():
        if inp.name not in raw_feed:
            raise KeyError(f"ONNX input '{inp.name}' not found in feed keys {list(raw_feed.keys())}")
        target_dtype = _ort_type_to_numpy(inp.type)
        arr = raw_feed[inp.name]
        if arr.dtype != target_dtype:
            arr = arr.astype(target_dtype)
        feed[inp.name] = np.ascontiguousarray(arr)
    return feed


def _run_ort_session_iobinding(
    session: Any,
    cuda_tensors: dict[str, torch.Tensor],
    preallocated_outputs: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Run an ORT session via GPU zero-copy IO binding using PyTorch CUDA tensors."""
    binding = session.io_binding()
    for inp in session.get_inputs():
        t = cuda_tensors[inp.name]
        target_dtype = _ort_type_to_torch(inp.type)
        if t.dtype != target_dtype:
            t = t.to(dtype=target_dtype)
        t = t.contiguous()
        cuda_tensors[inp.name] = t
        np_dtype = _ort_type_to_numpy(inp.type)
        binding.bind_input(
            name=inp.name,
            device_type="cuda",
            device_id=0,
            element_type=np_dtype,
            shape=tuple(t.shape),
            buffer_ptr=t.data_ptr(),
        )

    outputs: dict[str, torch.Tensor] = {}
    for out in session.get_outputs():
        if preallocated_outputs is not None and out.name in preallocated_outputs:
            out_t = preallocated_outputs[out.name]
            binding.bind_output(
                name=out.name,
                device_type="cuda",
                device_id=0,
                element_type=_ort_type_to_numpy(out.type),
                shape=tuple(out_t.shape),
                buffer_ptr=out_t.data_ptr(),
            )
            outputs[out.name] = out_t
        else:
            binding.bind_output(out.name, "cuda", 0)

    _sync_device("cuda")
    binding.synchronize_inputs()
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()

    if preallocated_outputs is None:
        ort_outs = binding.get_outputs()
        for out_meta, ort_val in zip(session.get_outputs(), ort_outs, strict=True):
            # Convert ORT CUDA OrtValue to PyTorch CUDA tensor via dlpack or numpy fallback
            try:
                outputs[out_meta.name] = torch.from_dlpack(ort_val.to_dlpack())
            except Exception:
                outputs[out_meta.name] = torch.from_numpy(ort_val.numpy()).to("cuda")
    return outputs


def evaluate_onnx_directory(
    onnx_dir: Path,
    ref_batches: list[dict[str, Any]],
    precision: str,
    device: str,
    provider_choice: str,
    use_io_binding: bool,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    import onnxruntime as ort

    if use_io_binding and device != "cuda":
        raise ValueError("--io-binding requires --device cuda")

    onnx_paths = _resolve_onnx_files(onnx_dir)
    providers = _make_ort_providers(device, provider_choice)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    disk_size_mb = _dir_onnx_size_mb(onnx_dir)

    with ResourceMonitor(track_gpu=(device == "cuda")) as monitor:
        sess_vision = ort.InferenceSession(str(onnx_paths["vision"]), sess_options=sess_opts, providers=providers)
        sess_prefix = ort.InferenceSession(str(onnx_paths["prefix"]), sess_options=sess_opts, providers=providers)
        sess_expert = ort.InferenceSession(str(onnx_paths["expert"]), sess_options=sess_opts, providers=providers)
        active_providers = sess_vision.get_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in active_providers and "TensorrtExecutionProvider" not in active_providers:
            raise RuntimeError(
                f"CUDA requested, but ORT fell back to CPU-only providers: {active_providers}"
            )
        rss_after_load_mb = psutil.Process(os.getpid()).memory_info().rss / (1024.0**2)

        vision_out_name = sess_vision.get_outputs()[0].name
        prefix_out_names = [o.name for o in sess_prefix.get_outputs()]
        expert_out_name = sess_expert.get_outputs()[0].name

        def _run_numpy_batch(b: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
            bsize = b["state"].shape[0]
            pv = b["pixel_values"].numpy()
            img_masks = b["img_masks"].numpy()
            lang_tokens = b["lang_tokens"].numpy()
            lang_masks = b["lang_masks"].numpy()
            state = b["state"].numpy()
            x_1 = b["x_1"].numpy()

            t0 = time.perf_counter()
            vis_feed = _cast_feed(sess_vision, {"pixel_values": pv})
            vis_outs = sess_vision.run(None, vis_feed)
            img_emb = vis_outs[0]
            t1 = time.perf_counter()

            img_emb_seq = img_emb.reshape(bsize, NUM_CAMERAS * 64, 960)
            pref_feed = _cast_feed(
                sess_prefix,
                {
                    "image_embeddings": img_emb_seq,
                    "img_masks": img_masks,
                    "lang_tokens": lang_tokens,
                    "lang_masks": lang_masks,
                    "state": state,
                },
            )
            pref_outs = sess_prefix.run(None, pref_feed)
            pref_map = dict(zip(prefix_out_names, pref_outs, strict=True))
            past_keys = pref_map["past_keys"]
            past_values = pref_map["past_values"]
            prefix_pad_masks = pref_map["prefix_pad_masks"]
            t2 = time.perf_counter()

            x_t = x_1.copy()
            dt = -1.0 / NUM_EULER_STEPS
            for step in range(NUM_EULER_STEPS):
                t_val = np.full((bsize,), 1.0 + step * dt, dtype=np.float32)
                exp_feed = _cast_feed(
                    sess_expert,
                    {
                        "x_t": x_t,
                        "timestep": t_val,
                        "past_keys": past_keys,
                        "past_values": past_values,
                        "prefix_pad_masks": prefix_pad_masks,
                    },
                )
                v_t = sess_expert.run(None, exp_feed)[0].astype(np.float32)
                x_t = x_t + dt * v_t
            t3 = time.perf_counter()

            # Isolated single-step velocity against reference prefix KV cache
            ref_exp_feed = _cast_feed(
                sess_expert,
                {
                    "x_t": x_1,
                    "timestep": np.ones((bsize,), dtype=np.float32),
                    "past_keys": b["ref_past_keys"].numpy(),
                    "past_values": b["ref_past_values"].numpy(),
                    "prefix_pad_masks": b["ref_prefix_pad_masks"].numpy(),
                },
            )
            isolated_v_t = sess_expert.run(None, ref_exp_feed)[0].astype(np.float32)

            return (
                {
                    "image_embeddings": img_emb.astype(np.float32),
                    "past_keys": past_keys.astype(np.float32),
                    "past_values": past_values.astype(np.float32),
                    "prefix_pad_masks": prefix_pad_masks,
                    "isolated_v_t": isolated_v_t,
                    "actions": x_t.astype(np.float32),
                },
                {
                    "vision_ms": (t1 - t0) * 1000.0,
                    "prefix_ms": (t2 - t1) * 1000.0,
                    "expert_10steps_ms": (t3 - t2) * 1000.0,
                    "total_ms": (t3 - t0) * 1000.0,
                },
            )

        def _prepare_cuda_batch(b: dict[str, Any]) -> dict[str, torch.Tensor]:
            return {
                "pixel_values": b["pixel_values"].to("cuda").contiguous(),
                "img_masks": b["img_masks"].to("cuda").contiguous(),
                "lang_tokens": b["lang_tokens"].to("cuda").contiguous(),
                "lang_masks": b["lang_masks"].to("cuda").contiguous(),
                "state": b["state"].to("cuda").contiguous(),
                "x_1": b["x_1"].to("cuda").contiguous(),
                "ref_past_keys": b["ref_past_keys"].to("cuda").contiguous(),
                "ref_past_values": b["ref_past_values"].to("cuda").contiguous(),
                "ref_prefix_pad_masks": b["ref_prefix_pad_masks"].to("cuda").contiguous(),
            }

        def _run_iobinding_batch(
            cb: dict[str, torch.Tensor], compute_isolated_vt: bool = False
        ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
            bsize = cb["state"].shape[0]
            vis_out_dtype = _ort_type_to_torch(sess_vision.get_outputs()[0].type)
            pref_out_dtypes = {o.name: _ort_type_to_torch(o.type) for o in sess_prefix.get_outputs()}
            exp_out_dtype = _ort_type_to_torch(sess_expert.get_outputs()[0].type)

            img_emb_buf = torch.empty((bsize * NUM_CAMERAS, 64, 960), dtype=vis_out_dtype, device="cuda")
            past_k_buf = torch.empty(
                (NUM_VLM_LAYERS, bsize, NUM_KV_HEADS, PREFIX_LEN, HEAD_DIM),
                dtype=pref_out_dtypes["past_keys"],
                device="cuda",
            )
            past_v_buf = torch.empty(
                (NUM_VLM_LAYERS, bsize, NUM_KV_HEADS, PREFIX_LEN, HEAD_DIM),
                dtype=pref_out_dtypes["past_values"],
                device="cuda",
            )
            pad_mask_buf = torch.empty((bsize, PREFIX_LEN), dtype=torch.bool, device="cuda")
            vt_buf = torch.empty((bsize, CHUNK_SIZE, MAX_ACTION_DIM), dtype=exp_out_dtype, device="cuda")

            _sync_device("cuda")
            t0 = time.perf_counter()
            vis_out = _run_ort_session_iobinding(
                sess_vision,
                {"pixel_values": cb["pixel_values"]},
                {vision_out_name: img_emb_buf},
            )
            _sync_device("cuda")
            t1 = time.perf_counter()

            img_emb_seq = vis_out[vision_out_name].reshape(bsize, NUM_CAMERAS * 64, 960)
            pref_out = _run_ort_session_iobinding(
                sess_prefix,
                {
                    "image_embeddings": img_emb_seq,
                    "img_masks": cb["img_masks"],
                    "lang_tokens": cb["lang_tokens"],
                    "lang_masks": cb["lang_masks"],
                    "state": cb["state"],
                },
                {
                    "past_keys": past_k_buf,
                    "past_values": past_v_buf,
                    "prefix_pad_masks": pad_mask_buf,
                },
            )
            _sync_device("cuda")
            t2 = time.perf_counter()

            x_t = cb["x_1"].clone()
            dt = -1.0 / NUM_EULER_STEPS
            for step in range(NUM_EULER_STEPS):
                t_val = torch.full((bsize,), 1.0 + step * dt, dtype=torch.float32, device="cuda")
                exp_out = _run_ort_session_iobinding(
                    sess_expert,
                    {
                        "x_t": x_t,
                        "timestep": t_val,
                        "past_keys": pref_out["past_keys"],
                        "past_values": pref_out["past_values"],
                        "prefix_pad_masks": pref_out["prefix_pad_masks"],
                    },
                    {expert_out_name: vt_buf},
                )
                x_t = x_t + dt * exp_out[expert_out_name].float()
            _sync_device("cuda")
            t3 = time.perf_counter()

            timings = {
                "vision_ms": (t1 - t0) * 1000.0,
                "prefix_ms": (t2 - t1) * 1000.0,
                "expert_10steps_ms": (t3 - t2) * 1000.0,
                "total_ms": (t3 - t0) * 1000.0,
            }
            if not compute_isolated_vt:
                return {}, timings

            iso_out = _run_ort_session_iobinding(
                sess_expert,
                {
                    "x_t": cb["x_1"],
                    "timestep": torch.ones((bsize,), dtype=torch.float32, device="cuda"),
                    "past_keys": cb["ref_past_keys"],
                    "past_values": cb["ref_past_values"],
                    "prefix_pad_masks": cb["ref_prefix_pad_masks"],
                },
                {expert_out_name: vt_buf},
            )
            return (
                {
                    "image_embeddings": img_emb_buf.float().cpu().numpy(),
                    "past_keys": past_k_buf.float().cpu().numpy(),
                    "past_values": past_v_buf.float().cpu().numpy(),
                    "prefix_pad_masks": pad_mask_buf.cpu().numpy(),
                    "isolated_v_t": iso_out[expert_out_name].float().cpu().numpy(),
                    "actions": x_t.float().cpu().numpy(),
                },
                timings,
            )

        # Accuracy evaluation across all reference batches
        acc_lists: dict[str, list[AccuracyMetrics]] = {
            "vision_image_embeddings": [],
            "prefix_past_keys": [],
            "prefix_past_values": [],
            "expert_single_step_v_t": [],
            "actions_10step": [],
        }
        pad_mask_exact_matches = []

        if use_io_binding:
            cuda_batches = [_prepare_cuda_batch(b) for b in ref_batches]
            for b, cb in zip(ref_batches, cuda_batches, strict=True):
                outs, _ = _run_iobinding_batch(cb, compute_isolated_vt=True)
                acc_lists["vision_image_embeddings"].append(
                    compute_accuracy_metrics(outs["image_embeddings"], b["ref_image_embeddings"].numpy())
                )
                acc_lists["prefix_past_keys"].append(
                    compute_accuracy_metrics(outs["past_keys"], b["ref_past_keys"].numpy())
                )
                acc_lists["prefix_past_values"].append(
                    compute_accuracy_metrics(outs["past_values"], b["ref_past_values"].numpy())
                )
                acc_lists["expert_single_step_v_t"].append(
                    compute_accuracy_metrics(outs["isolated_v_t"], b["ref_v_t"].numpy())
                )
                acc_lists["actions_10step"].append(
                    compute_accuracy_metrics(outs["actions"], b["ref_actions"].numpy())
                )
                pad_mask_exact_matches.append(
                    bool(np.array_equal(outs["prefix_pad_masks"], b["ref_prefix_pad_masks"].numpy()))
                )

            for w in range(warmup):
                _run_iobinding_batch(cuda_batches[w % len(cuda_batches)], compute_isolated_vt=False)

            vision_times, prefix_times, expert_times, total_times = [], [], [], []
            for r in range(runs):
                _, tm = _run_iobinding_batch(cuda_batches[r % len(cuda_batches)], compute_isolated_vt=False)
                vision_times.append(tm["vision_ms"])
                prefix_times.append(tm["prefix_ms"])
                expert_times.append(tm["expert_10steps_ms"])
                total_times.append(tm["total_ms"])
            del cuda_batches
        else:
            for b in ref_batches:
                outs, _ = _run_numpy_batch(b)
                acc_lists["vision_image_embeddings"].append(
                    compute_accuracy_metrics(outs["image_embeddings"], b["ref_image_embeddings"].numpy())
                )
                acc_lists["prefix_past_keys"].append(
                    compute_accuracy_metrics(outs["past_keys"], b["ref_past_keys"].numpy())
                )
                acc_lists["prefix_past_values"].append(
                    compute_accuracy_metrics(outs["past_values"], b["ref_past_values"].numpy())
                )
                acc_lists["expert_single_step_v_t"].append(
                    compute_accuracy_metrics(outs["isolated_v_t"], b["ref_v_t"].numpy())
                )
                acc_lists["actions_10step"].append(
                    compute_accuracy_metrics(outs["actions"], b["ref_actions"].numpy())
                )
                pad_mask_exact_matches.append(
                    bool(np.array_equal(outs["prefix_pad_masks"], b["ref_prefix_pad_masks"].numpy()))
                )

            for w in range(warmup):
                _run_numpy_batch(ref_batches[w % len(ref_batches)])

            vision_times, prefix_times, expert_times, total_times = [], [], [], []
            for r in range(runs):
                _, tm = _run_numpy_batch(ref_batches[r % len(ref_batches)])
                vision_times.append(tm["vision_ms"])
                prefix_times.append(tm["prefix_ms"])
                expert_times.append(tm["expert_10steps_ms"])
                total_times.append(tm["total_ms"])

        active_providers = sess_vision.get_providers()
        del sess_vision, sess_prefix, sess_expert
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    mem_stats = monitor.summary()
    mem_stats["rss_after_load_mb"] = round(rss_after_load_mb, 2)
    mem_stats["onnx_disk_size_mb"] = disk_size_mb

    lat_summary = summarize_latencies(total_times)
    chunks_per_sec = round(1000.0 / max(lat_summary["mean_ms"], 1e-6), 3)
    actions_per_sec = round(chunks_per_sec * CHUNK_SIZE, 3)

    acc_summary: dict[str, Any] = {}
    for key, items in acc_lists.items():
        acc_summary[key] = {
            "max_abs_err": max(m.max_abs_err for m in items),
            "rel_max_abs_err": max(m.rel_max_abs_err for m in items),
            "mae": float(np.mean([m.mae for m in items])),
            "rmse": float(np.mean([m.rmse for m in items])),
            "cosine_sim": float(np.mean([m.cosine_sim for m in items])),
            "min_cosine_sim": min(m.cosine_sim for m in items),
        }
    acc_summary["prefix_pad_masks_exact_match"] = all(pad_mask_exact_matches)

    thresholds = ACCEPTANCE_THRESHOLDS[precision]
    act_acc = acc_summary["actions_10step"]
    passed_accuracy = act_acc["min_cosine_sim"] >= thresholds["actions_cosine_sim"]
    if "actions_max_abs_err" in thresholds:
        passed_accuracy = passed_accuracy and (act_acc["max_abs_err"] < thresholds["actions_max_abs_err"])

    est_runtime_mem_mb = disk_size_mb + (mem_stats["nvml_gpu_delta_mb"] or mem_stats["rss_delta_mb"] or 0.0) * 0.25 + 512.0
    jetson_budget_ok = disk_size_mb <= 2200.0 and (
        mem_stats["nvml_gpu_delta_mb"] is None or mem_stats["nvml_gpu_delta_mb"] <= JETSON_SAFE_BUDGET_MB
    )

    return {
        "backend": f"onnx_{precision}_{device}",
        "onnx_dir": str(onnx_dir),
        "onnx_files": {k: str(v.name) for k, v in onnx_paths.items()},
        "precision": precision,
        "device": device,
        "providers": active_providers,
        "io_binding": use_io_binding,
        "warmup_runs": warmup,
        "timed_runs": runs,
        "latency": lat_summary,
        "breakdown_ms": {
            "vision_mean_ms": round(float(np.mean(vision_times)), 3),
            "prefix_mean_ms": round(float(np.mean(prefix_times)), 3),
            "expert_10steps_mean_ms": round(float(np.mean(expert_times)), 3),
            "expert_per_step_mean_ms": round(float(np.mean(expert_times)) / NUM_EULER_STEPS, 3),
        },
        "throughput": {
            "chunks_per_sec": chunks_per_sec,
            "control_actions_per_sec": actions_per_sec,
        },
        "memory": mem_stats,
        "accuracy_vs_fp32_ref": acc_summary,
        "acceptance": {
            "thresholds": thresholds,
            "passed_accuracy": passed_accuracy,
            "jetson_orin_nano_8gb_ok": jetson_budget_ok,
            "jetson_mem_budget_mb": JETSON_ORIN_NANO_MEM_MB,
            "jetson_safe_limit_mb": JETSON_SAFE_BUDGET_MB,
            "onnx_disk_size_mb": disk_size_mb,
            "est_jetson_total_mb": round(est_runtime_mem_mb, 2),
        },
    }


def _compute_baseline_comparison(onnx_res: dict[str, Any], baseline_data: dict[str, Any]) -> dict[str, Any]:
    comparisons = {}
    device = onnx_res["device"]
    onnx_mean_ms = onnx_res["latency"]["mean_ms"]
    onnx_rss = onnx_res["memory"]["rss_peak_mb"]
    onnx_vram = onnx_res["memory"].get("nvml_gpu_peak_mb")

    for name, b_res in baseline_data.get("benchmarks", {}).items():
        if b_res.get("device") != device:
            continue
        b_mean = b_res["latency"]["mean_ms"]
        b_rss = b_res["memory"]["rss_peak_mb"]
        b_vram = b_res["memory"].get("torch_peak_vram_allocated_mb") or b_res["memory"].get("nvml_gpu_peak_mb")
        comparisons[name] = {
            "baseline_mean_ms": b_mean,
            "onnx_mean_ms": onnx_mean_ms,
            "speedup_vs_baseline": round(b_mean / max(onnx_mean_ms, 1e-6), 3),
            "rss_reduction_mb": round(b_rss - onnx_rss, 2),
            "vram_reduction_mb": round(b_vram - onnx_vram, 2) if (b_vram and onnx_vram) else None,
        }
    return comparisons


def _print_summary_table(report: dict[str, Any]) -> None:
    print("\n=== Track B Robotics Evaluation Summary ===")
    print(f"Model: {MODEL_REPO_ID} @ {MODEL_REVISION[:12]} | Batches: {report['num_batches']} | B={report['batch_size']}")
    if "benchmarks" in report:
        header = f"{'Backend':<22} {'Mean(ms)':>9} {'p50(ms)':>9} {'p95(ms)':>9} {'Chunk/s':>8} {'Act/s':>8} {'VRAM(MB)':>9} {'RSS(MB)':>9} {'ActCos':>10} {'ActMaxErr':>11}"
        print(header)
        print("-" * len(header))
        for key, res in report["benchmarks"].items():
            vram = res["memory"].get("torch_peak_vram_allocated_mb") or res["memory"].get("nvml_gpu_peak_mb") or 0.0
            act = res["accuracy_vs_fp32_ref"]["actions_10step"]
            print(
                f"{key:<22} "
                f"{res['latency']['mean_ms']:>9.2f} "
                f"{res['latency']['p50_ms']:>9.2f} "
                f"{res['latency']['p95_ms']:>9.2f} "
                f"{res['throughput']['chunks_per_sec']:>8.2f} "
                f"{res['throughput']['control_actions_per_sec']:>8.1f} "
                f"{vram:>9.1f} "
                f"{res['memory']['rss_peak_mb']:>9.1f} "
                f"{act['cosine_sim']:>10.6f} "
                f"{act['max_abs_err']:>11.3e}"
            )
    if "onnx_eval" in report:
        res = report["onnx_eval"]
        vram = res["memory"].get("nvml_gpu_delta_mb") or res["memory"].get("nvml_gpu_peak_mb") or 0.0
        print(
            f"\nONNX ({res['precision']}, {res['device']}, io_binding={res['io_binding']}): "
            f"mean={res['latency']['mean_ms']:.2f} ms (p50={res['latency']['p50_ms']:.2f}, p95={res['latency']['p95_ms']:.2f}), "
            f"{res['throughput']['chunks_per_sec']:.2f} chunks/s ({res['throughput']['control_actions_per_sec']:.1f} act/s), "
            f"disk={res['memory']['onnx_disk_size_mb']:.1f} MB, GPU_delta={vram:.1f} MB, RSS_peak={res['memory']['rss_peak_mb']:.1f} MB"
        )
        print("Breakdown (ms):", res["breakdown_ms"])
        print("Accuracy vs PyTorch FP32 reference:")
        for sg, m in res["accuracy_vs_fp32_ref"].items():
            if isinstance(m, dict):
                print(
                    f"  {sg:<26} MaxAbsErr={m['max_abs_err']:.3e} (Rel={m['rel_max_abs_err']:.3e}) "
                    f"MAE={m['mae']:.3e} RMSE={m['rmse']:.3e} Cos={m['cosine_sim']:.6f}"
                )
            else:
                print(f"  {sg:<26} {m}")
        print("Acceptance:", res["acceptance"])
        if report.get("baseline_comparison"):
            print("Speedup vs Baseline:", report["baseline_comparison"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked Track B SmolVLA evaluation harness.")
    parser.add_argument("--onnx-dir", type=Path, default=None, help="Directory with exported ONNX subgraphs.")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--provider", choices=["auto", "cuda", "tensorrt", "cpu"], default="auto")
    parser.add_argument("--io-binding", action="store_true", help="Use ORT GPU IO binding for zero-copy chaining.")
    parser.add_argument("--baseline-only", action="store_true", help="Run PyTorch FP32/BF16 baselines & generate ref cache.")
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON, help="Path to baseline JSON.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON report path.")
    parser.add_argument("--num-batches", type=int, default=DEFAULT_NUM_BATCHES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gpu-warmup", type=int, default=5)
    parser.add_argument("--gpu-runs", type=int, default=20)
    parser.add_argument("--cpu-warmup", type=int, default=1)
    parser.add_argument("--cpu-runs", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.baseline_only and args.onnx_dir is None:
        raise SystemExit("Error: specify either --baseline-only or --onnx-dir <path>.")

    ref_payload = load_or_create_reference_batches(
        cache_path=REF_CACHE_PATH,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        seeds=DEFAULT_SEEDS,
    )
    ref_batches = ref_payload["batches"]

    report: dict[str, Any] = {
        "model_repo_id": MODEL_REPO_ID,
        "model_revision": MODEL_REVISION,
        "ref_cache_path": str(REF_CACHE_PATH),
        "num_batches": len(ref_batches),
        "batch_size": args.batch_size,
        "seeds_used": [b["seed"] for b in ref_batches],
    }

    if args.baseline_only:
        benchmarks: dict[str, Any] = {}
        if torch.cuda.is_available():
            benchmarks["pytorch_fp32_cuda"] = benchmark_pytorch_policy(
                ref_batches, mode="fp32", device="cuda", warmup=args.gpu_warmup, runs=args.gpu_runs
            )
            benchmarks["pytorch_bf16_cuda"] = benchmark_pytorch_policy(
                ref_batches, mode="bf16", device="cuda", warmup=args.gpu_warmup, runs=args.gpu_runs
            )
        if args.cpu_runs > 0:
            benchmarks["pytorch_fp32_cpu"] = benchmark_pytorch_policy(
                ref_batches, mode="fp32", device="cpu", warmup=args.cpu_warmup, runs=args.cpu_runs
            )
        report["benchmarks"] = benchmarks
        out_path = args.output_json or DEFAULT_BASELINE_JSON
    else:
        warmup = args.gpu_warmup if args.device == "cuda" else args.cpu_warmup
        runs = args.gpu_runs if args.device == "cuda" else args.cpu_runs
        onnx_res = evaluate_onnx_directory(
            onnx_dir=args.onnx_dir,
            ref_batches=ref_batches,
            precision=args.precision,
            device=args.device,
            provider_choice=args.provider,
            use_io_binding=args.io_binding,
            warmup=warmup,
            runs=runs,
        )
        report["onnx_eval"] = onnx_res
        if args.baseline_json and args.baseline_json.exists():
            with open(args.baseline_json, encoding="utf-8") as f:
                baseline_data = json.load(f)
            report["baseline_comparison"] = _compute_baseline_comparison(onnx_res, baseline_data)
        out_path = args.output_json

    _print_summary_table(report)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved evaluation report to {out_path}")

    if "onnx_eval" in report and not report["onnx_eval"]["acceptance"]["passed_accuracy"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
