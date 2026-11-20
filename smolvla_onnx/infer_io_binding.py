#!/usr/bin/env python3
"""Standalone, PyTorch-free ONNX Runtime I/O-Binding inference engine for SmolVLA.

Pre-allocates persistent CUDA (or CPU) device buffers for all intermediate activations
(`image_embeddings`, `past_keys`, `past_values`, `prefix_pad_masks`, `x_t`, `timestep`,
`v_t`) and executes the entire vision -> VLM prefix -> 10-step Euler flow-matching loop
with ZERO host-device copies and ZERO per-step allocations between denoising steps.

Requires ONLY `numpy` and `onnxruntime` (plus `onnx` for building the 2-node in-memory
GPU Euler-update micro-graph; falls back cleanly if `onnx` is absent).

Example usage:
    from infer_io_binding import SmolVLAOnnxRunner

    runner = SmolVLAOnnxRunner("artifacts/smolvla_onnx_fp16", device="cuda")
    actions = runner.predict_action_chunk(
        pixel_values=pixel_values,   # [3, 3, 512, 512] float32 in [-1, 1]
        img_masks=img_masks,         # [1, 3] bool
        lang_tokens=lang_tokens,     # [1, 48] int64
        lang_masks=lang_masks,       # [1, 48] bool
        state=state,                 # [1, 32] float32
        noise=x_1,                   # [1, 50, 32] float32 (optional)
    )

CLI benchmark:
    python infer_io_binding.py --onnx-dir artifacts/smolvla_onnx_fp16 --benchmark
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

NUM_CAMERAS = 3
IMAGE_SIZE = 512
VISION_OUT_TOKENS = 64
VLM_HIDDEN = 960
LANG_MAX_LEN = 48
MAX_STATE_DIM = 32
MAX_ACTION_DIM = 32
CHUNK_SIZE = 50
NUM_EULER_STEPS = 10
NUM_VLM_LAYERS = 16
NUM_KV_HEADS = 5
HEAD_DIM = 64
PREFIX_LEN = 241


def _ort_type_to_numpy(ort_type: str) -> np.dtype:
    if "float16" in ort_type:
        return np.dtype(np.float16)
    if "float" in ort_type:
        return np.dtype(np.float32)
    if "int64" in ort_type:
        return np.dtype(np.int64)
    if "int32" in ort_type:
        return np.dtype(np.int32)
    if "bool" in ort_type:
        return np.dtype(np.bool_)
    raise ValueError(f"Unsupported ONNX Runtime tensor type: {ort_type}")


def _build_euler_step_onnx_bytes(
    x_dtype: np.dtype, v_dtype: np.dtype, chunk_size: int = CHUNK_SIZE, action_dim: int = MAX_ACTION_DIM
) -> bytes | None:
    """Build a 2-node in-memory ONNX graph `x_next = x_t + dt * v_t` for zero-copy GPU updates."""
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return None

    x_tp = TensorProto.FLOAT16 if x_dtype == np.float16 else TensorProto.FLOAT
    v_tp = TensorProto.FLOAT16 if v_dtype == np.float16 else TensorProto.FLOAT

    nodes = []
    v_in = "v_t"
    if v_tp != x_tp:
        nodes.append(helper.make_node("Cast", inputs=["v_t"], outputs=["v_cast"], to=x_tp))
        v_in = "v_cast"

    nodes.append(helper.make_node("Mul", inputs=[v_in, "dt"], outputs=["delta"]))
    nodes.append(helper.make_node("Add", inputs=["x_t", "delta"], outputs=["x_next"]))

    graph = helper.make_graph(
        nodes=nodes,
        name="euler_update_step",
        inputs=[
            helper.make_tensor_value_info("x_t", x_tp, [1, chunk_size, action_dim]),
            helper.make_tensor_value_info("v_t", v_tp, [1, chunk_size, action_dim]),
            helper.make_tensor_value_info("dt", x_tp, [1]),
        ],
        outputs=[
            helper.make_tensor_value_info("x_next", x_tp, [1, chunk_size, action_dim]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model.SerializeToString()


class SmolVLAOnnxRunner:
    """Zero-copy ONNX Runtime I/O-Binding runner for the 3 static SmolVLA ONNX subgraphs."""

    def __init__(
        self,
        onnx_dir: str | Path,
        device: str = "cuda",
        provider: str = "auto",
        device_id: int = 0,
        num_steps: int = NUM_EULER_STEPS,
        intra_op_num_threads: int = 0,
    ) -> None:
        self.onnx_dir = Path(onnx_dir)
        self.device = device
        self.device_id = device_id
        self.num_steps = num_steps
        self.dt = -1.0 / float(num_steps)

        if device == "cuda" and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()

        providers = self._resolve_providers(device, provider, device_id)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_num_threads > 0:
            sess_opts.intra_op_num_threads = intra_op_num_threads

        self.sess_vision = ort.InferenceSession(
            str(self.onnx_dir / "smolvla_vision_encoder.onnx"),
            sess_options=sess_opts,
            providers=providers,
        )
        self.sess_prefix = ort.InferenceSession(
            str(self.onnx_dir / "smolvla_vlm_prefix.onnx"),
            sess_options=sess_opts,
            providers=providers,
        )
        self.sess_expert = ort.InferenceSession(
            str(self.onnx_dir / "smolvla_action_expert_step.onnx"),
            sess_options=sess_opts,
            providers=providers,
        )
        self.active_providers = self.sess_vision.get_providers()

        if device == "cuda" and not any(
            p in self.active_providers for p in ("CUDAExecutionProvider", "TensorrtExecutionProvider")
        ):
            raise RuntimeError(
                f"CUDA requested, but ORT initialized with CPU-only providers: {self.active_providers}"
            )

        # Inspect exact tensor dtypes from the loaded ONNX models
        self._vis_in_dtype = _ort_type_to_numpy(self.sess_vision.get_inputs()[0].type)
        self._vis_out_dtype = _ort_type_to_numpy(self.sess_vision.get_outputs()[0].type)

        pref_in = {i.name: _ort_type_to_numpy(i.type) for i in self.sess_prefix.get_inputs()}
        pref_out = {o.name: _ort_type_to_numpy(o.type) for o in self.sess_prefix.get_outputs()}
        exp_in = {i.name: _ort_type_to_numpy(i.type) for i in self.sess_expert.get_inputs()}
        exp_out = {o.name: _ort_type_to_numpy(o.type) for o in self.sess_expert.get_outputs()}

        self._state_dtype = pref_in["state"]
        self._pk_dtype = pref_out["past_keys"]
        self._pv_dtype = pref_out["past_values"]
        self._xt_dtype = exp_in["x_t"]
        self._t_dtype = exp_in["timestep"]
        self._vt_dtype = exp_out["v_t"]

        self._use_cuda_iobind = device == "cuda"
        self._sess_euler: ort.InferenceSession | None = None
        if self._use_cuda_iobind:
            self._init_cuda_persistent_buffers(sess_opts, providers)

    @staticmethod
    def _resolve_providers(device: str, provider: str, device_id: int) -> list[Any]:
        avail = ort.get_available_providers()
        if device == "cpu" or provider == "cpu":
            return ["CPUExecutionProvider"]
        if provider == "tensorrt":
            if "TensorrtExecutionProvider" not in avail:
                raise RuntimeError(f"TensorrtExecutionProvider not available: {avail}")
            return [
                ("TensorrtExecutionProvider", {"device_id": device_id}),
                ("CUDAExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ]
        if "CUDAExecutionProvider" not in avail:
            raise RuntimeError(f"CUDAExecutionProvider not available: {avail}")
        return [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]

    def _alloc_cuda(self, shape: tuple[int, ...], dtype: np.dtype) -> ort.OrtValue:
        return ort.OrtValue.ortvalue_from_shape_and_type(list(shape), dtype, "cuda", self.device_id)

    def _init_cuda_persistent_buffers(self, sess_opts: ort.SessionOptions, providers: list[Any]) -> None:
        """Allocate persistent CUDA OrtValue buffers and pre-bind all 3 subgraphs + Euler loop."""
        # Persistent device buffers
        self._buf_pixel_values = self._alloc_cuda((NUM_CAMERAS, 3, IMAGE_SIZE, IMAGE_SIZE), self._vis_in_dtype)
        self._buf_image_embs = self._alloc_cuda((NUM_CAMERAS, VISION_OUT_TOKENS, VLM_HIDDEN), self._vis_out_dtype)
        self._buf_img_masks = self._alloc_cuda((1, NUM_CAMERAS), np.dtype(np.bool_))
        self._buf_lang_tokens = self._alloc_cuda((1, LANG_MAX_LEN), np.dtype(np.int64))
        self._buf_lang_masks = self._alloc_cuda((1, LANG_MAX_LEN), np.dtype(np.bool_))
        self._buf_state = self._alloc_cuda((1, MAX_STATE_DIM), self._state_dtype)

        self._buf_past_keys = self._alloc_cuda(
            (NUM_VLM_LAYERS, 1, NUM_KV_HEADS, PREFIX_LEN, HEAD_DIM), self._pk_dtype
        )
        self._buf_past_values = self._alloc_cuda(
            (NUM_VLM_LAYERS, 1, NUM_KV_HEADS, PREFIX_LEN, HEAD_DIM), self._pv_dtype
        )
        self._buf_prefix_pad_masks = self._alloc_cuda((1, PREFIX_LEN), np.dtype(np.bool_))

        # Ping-pong buffers for x_t so Euler integration stays 100% on GPU
        self._buf_x = [
            self._alloc_cuda((1, CHUNK_SIZE, MAX_ACTION_DIM), self._xt_dtype),
            self._alloc_cuda((1, CHUNK_SIZE, MAX_ACTION_DIM), self._xt_dtype),
        ]
        self._buf_v_t = self._alloc_cuda((1, CHUNK_SIZE, MAX_ACTION_DIM), self._vt_dtype)

        # Pre-upload constant timesteps [1.0, 0.9, ..., 0.1] and dt=-0.1 to GPU once
        self._buf_timesteps: list[ort.OrtValue] = []
        for step in range(self.num_steps):
            t_arr = np.array([1.0 + step * self.dt], dtype=self._t_dtype)
            self._buf_timesteps.append(ort.OrtValue.ortvalue_from_numpy(t_arr, "cuda", self.device_id))

        dt_arr = np.array([self.dt], dtype=self._xt_dtype)
        self._buf_dt = ort.OrtValue.ortvalue_from_numpy(dt_arr, "cuda", self.device_id)

        # Build tiny 2-node GPU Euler update session (zero host-device copies between steps)
        euler_bytes = _build_euler_step_onnx_bytes(self._xt_dtype, self._vt_dtype)
        if euler_bytes is not None:
            cuda_only = [("CUDAExecutionProvider", {"device_id": self.device_id})]
            self._sess_euler = ort.InferenceSession(euler_bytes, sess_options=sess_opts, providers=cuda_only)

        # Pre-bind Subgraph 1: Vision Encoder
        self._bind_vision = self.sess_vision.io_binding()
        self._bind_vision.bind_ortvalue_input("pixel_values", self._buf_pixel_values)
        self._bind_vision.bind_ortvalue_output("image_embeddings", self._buf_image_embs)

        # Pre-bind Subgraph 2: VLM Prefix (zero-copy view of image_embeddings [3,64,960] as [1,192,960])
        self._bind_prefix = self.sess_prefix.io_binding()
        self._bind_prefix.bind_input(
            name="image_embeddings",
            device_type="cuda",
            device_id=self.device_id,
            element_type=self._vis_out_dtype,
            shape=(1, NUM_CAMERAS * VISION_OUT_TOKENS, VLM_HIDDEN),
            buffer_ptr=self._buf_image_embs.data_ptr(),
        )
        self._bind_prefix.bind_ortvalue_input("img_masks", self._buf_img_masks)
        self._bind_prefix.bind_ortvalue_input("lang_tokens", self._buf_lang_tokens)
        self._bind_prefix.bind_ortvalue_input("lang_masks", self._buf_lang_masks)
        self._bind_prefix.bind_ortvalue_input("state", self._buf_state)
        self._bind_prefix.bind_ortvalue_output("past_keys", self._buf_past_keys)
        self._bind_prefix.bind_ortvalue_output("past_values", self._buf_past_values)
        self._bind_prefix.bind_ortvalue_output("prefix_pad_masks", self._buf_prefix_pad_masks)

        # Pre-bind Subgraph 3: 10 Action Expert step bindings + 2 Euler update ping-pong bindings
        self._bind_expert_steps: list[ort.IOBinding] = []
        for step in range(self.num_steps):
            src_idx = (step % 2) if self._sess_euler is not None else 0
            b = self.sess_expert.io_binding()
            b.bind_ortvalue_input("x_t", self._buf_x[src_idx])
            b.bind_ortvalue_input("timestep", self._buf_timesteps[step])
            b.bind_ortvalue_input("past_keys", self._buf_past_keys)
            b.bind_ortvalue_input("past_values", self._buf_past_values)
            b.bind_ortvalue_input("prefix_pad_masks", self._buf_prefix_pad_masks)
            b.bind_ortvalue_output("v_t", self._buf_v_t)
            self._bind_expert_steps.append(b)

        self._bind_euler_steps: list[ort.IOBinding] = []
        if self._sess_euler is not None:
            for parity in (0, 1):
                eb = self._sess_euler.io_binding()
                eb.bind_ortvalue_input("x_t", self._buf_x[parity])
                eb.bind_ortvalue_input("v_t", self._buf_v_t)
                eb.bind_ortvalue_input("dt", self._buf_dt)
                eb.bind_ortvalue_output("x_next", self._buf_x[1 - parity])
                self._bind_euler_steps.append(eb)

    def predict_action_chunk(
        self,
        pixel_values: np.ndarray,
        img_masks: np.ndarray,
        lang_tokens: np.ndarray,
        lang_masks: np.ndarray,
        state: np.ndarray,
        noise: np.ndarray | None = None,
        return_timings: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
        """Run full SmolVLA inference (Vision -> VLM Prefix -> 10-step Euler denoising).

        Returns:
            actions: float32 numpy array of shape `[1, 50, 32]`.
            timings (optional): per-stage latency dict in milliseconds.
        """
        if noise is None:
            noise = np.random.standard_normal((1, CHUNK_SIZE, MAX_ACTION_DIM)).astype(np.float32)

        if self._use_cuda_iobind:
            return self._predict_cuda_iobind(
                pixel_values, img_masks, lang_tokens, lang_masks, state, noise, return_timings
            )
        return self._predict_cpu(
            pixel_values, img_masks, lang_tokens, lang_masks, state, noise, return_timings
        )

    def _predict_cuda_iobind(
        self,
        pixel_values: np.ndarray,
        img_masks: np.ndarray,
        lang_tokens: np.ndarray,
        lang_masks: np.ndarray,
        state: np.ndarray,
        noise: np.ndarray,
        return_timings: bool,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
        # Stage 0: Single host-to-device copy of inputs into persistent CUDA buffers
        t_h2d_0 = time.perf_counter()
        self._buf_pixel_values.update_inplace(np.ascontiguousarray(pixel_values, dtype=self._vis_in_dtype))
        self._buf_img_masks.update_inplace(np.ascontiguousarray(img_masks, dtype=np.bool_))
        self._buf_lang_tokens.update_inplace(np.ascontiguousarray(lang_tokens, dtype=np.int64))
        self._buf_lang_masks.update_inplace(np.ascontiguousarray(lang_masks, dtype=np.bool_))
        self._buf_state.update_inplace(np.ascontiguousarray(state, dtype=self._state_dtype))
        self._buf_x[0].update_inplace(np.ascontiguousarray(noise, dtype=self._xt_dtype))
        self._bind_vision.synchronize_inputs()
        t0 = time.perf_counter()

        # Stage 1: Vision Encoder
        self.sess_vision.run_with_iobinding(self._bind_vision)
        self._bind_vision.synchronize_outputs()
        t1 = time.perf_counter()

        # Stage 2: VLM Prefix KV Prefill (reads image_embeddings directly on GPU)
        self.sess_prefix.run_with_iobinding(self._bind_prefix)
        self._bind_prefix.synchronize_outputs()
        t2 = time.perf_counter()

        # Stage 3: 10-step Euler flow-matching loop (ZERO host-device copies between steps!)
        if self._sess_euler is not None:
            for step in range(self.num_steps):
                self.sess_expert.run_with_iobinding(self._bind_expert_steps[step])
                self._bind_expert_steps[step].synchronize_outputs()
                self._sess_euler.run_with_iobinding(self._bind_euler_steps[step % 2])
                self._bind_euler_steps[step % 2].synchronize_outputs()
            final_buf = self._buf_x[self.num_steps % 2]
        else:
            x_t = np.ascontiguousarray(noise, dtype=np.float32)
            for step in range(self.num_steps):
                self._buf_x[0].update_inplace(np.ascontiguousarray(x_t, dtype=self._xt_dtype))
                self.sess_expert.run_with_iobinding(self._bind_expert_steps[step])
                self._bind_expert_steps[step].synchronize_outputs()
                v_t = self._buf_v_t.numpy().astype(np.float32)
                x_t = x_t + self.dt * v_t
            self._buf_x[0].update_inplace(np.ascontiguousarray(x_t, dtype=self._xt_dtype))
            final_buf = self._buf_x[0]
        t3 = time.perf_counter()

        # Single small device-to-host copy of final action chunk [1, 50, 32] (6.4 KB)
        actions = final_buf.numpy().astype(np.float32)
        t4 = time.perf_counter()

        if not return_timings:
            return actions
        return actions, {
            "h2d_ms": (t0 - t_h2d_0) * 1000.0,
            "vision_ms": (t1 - t0) * 1000.0,
            "prefix_ms": (t2 - t1) * 1000.0,
            "expert_10steps_ms": (t3 - t2) * 1000.0,
            "d2h_ms": (t4 - t3) * 1000.0,
            "device_compute_ms": (t3 - t0) * 1000.0,
            "total_e2e_ms": (t4 - t_h2d_0) * 1000.0,
        }

    def _predict_cpu(
        self,
        pixel_values: np.ndarray,
        img_masks: np.ndarray,
        lang_tokens: np.ndarray,
        lang_masks: np.ndarray,
        state: np.ndarray,
        noise: np.ndarray,
        return_timings: bool,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
        t0 = time.perf_counter()
        img_emb = self.sess_vision.run(
            None, {"pixel_values": np.ascontiguousarray(pixel_values, dtype=self._vis_in_dtype)}
        )[0]
        t1 = time.perf_counter()

        pref_outs = self.sess_prefix.run(
            None,
            {
                "image_embeddings": np.ascontiguousarray(
                    img_emb.reshape(1, NUM_CAMERAS * VISION_OUT_TOKENS, VLM_HIDDEN),
                    dtype=self._vis_out_dtype,
                ),
                "img_masks": np.ascontiguousarray(img_masks, dtype=np.bool_),
                "lang_tokens": np.ascontiguousarray(lang_tokens, dtype=np.int64),
                "lang_masks": np.ascontiguousarray(lang_masks, dtype=np.bool_),
                "state": np.ascontiguousarray(state, dtype=self._state_dtype),
            },
        )
        pref_names = [o.name for o in self.sess_prefix.get_outputs()]
        pref_map = dict(zip(pref_names, pref_outs, strict=True))
        t2 = time.perf_counter()

        x_t = np.ascontiguousarray(noise, dtype=np.float32)
        for step in range(self.num_steps):
            t_arr = np.array([1.0 + step * self.dt], dtype=self._t_dtype)
            v_t = self.sess_expert.run(
                None,
                {
                    "x_t": np.ascontiguousarray(x_t, dtype=self._xt_dtype),
                    "timestep": t_arr,
                    "past_keys": pref_map["past_keys"],
                    "past_values": pref_map["past_values"],
                    "prefix_pad_masks": pref_map["prefix_pad_masks"],
                },
            )[0].astype(np.float32)
            x_t = x_t + self.dt * v_t
        t3 = time.perf_counter()

        if not return_timings:
            return x_t
        return x_t, {
            "h2d_ms": 0.0,
            "vision_ms": (t1 - t0) * 1000.0,
            "prefix_ms": (t2 - t1) * 1000.0,
            "expert_10steps_ms": (t3 - t2) * 1000.0,
            "d2h_ms": 0.0,
            "device_compute_ms": (t3 - t0) * 1000.0,
            "total_e2e_ms": (t3 - t0) * 1000.0,
        }


def run_benchmark(
    onnx_dir: Path,
    device: str = "cuda",
    provider: str = "auto",
    warmup: int = 5,
    runs: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    runner = SmolVLAOnnxRunner(onnx_dir=onnx_dir, device=device, provider=provider)
    rng = np.random.default_rng(seed)

    pixel_values = rng.uniform(-1.0, 1.0, size=(NUM_CAMERAS, 3, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    img_masks = np.ones((1, NUM_CAMERAS), dtype=np.bool_)
    lang_tokens = np.full((1, LANG_MAX_LEN), 2, dtype=np.int64)
    lang_tokens[0, :10] = np.arange(100, 110, dtype=np.int64)
    lang_masks = np.zeros((1, LANG_MAX_LEN), dtype=np.bool_)
    lang_masks[0, :10] = True
    state = np.zeros((1, MAX_STATE_DIM), dtype=np.float32)
    state[0, :6] = rng.standard_normal(6).astype(np.float32) * 0.5
    noise = rng.standard_normal((1, CHUNK_SIZE, MAX_ACTION_DIM)).astype(np.float32)

    for _ in range(warmup):
        runner.predict_action_chunk(pixel_values, img_masks, lang_tokens, lang_masks, state, noise=noise)

    records: list[dict[str, float]] = []
    last_actions: np.ndarray | None = None
    for _ in range(runs):
        last_actions, tm = runner.predict_action_chunk(
            pixel_values, img_masks, lang_tokens, lang_masks, state, noise=noise, return_timings=True
        )
        records.append(tm)

    assert last_actions is not None and last_actions.shape == (1, CHUNK_SIZE, MAX_ACTION_DIM)
    assert np.all(np.isfinite(last_actions)), "Non-finite values in predicted action chunk!"

    compute_ms = np.array([r["device_compute_ms"] for r in records], dtype=np.float64)
    e2e_ms = np.array([r["total_e2e_ms"] for r in records], dtype=np.float64)
    mean_compute = float(np.mean(compute_ms))
    mean_e2e = float(np.mean(e2e_ms))

    summary = {
        "onnx_dir": str(onnx_dir),
        "device": device,
        "providers": runner.active_providers,
        "zero_copy_gpu_euler": bool(runner._use_cuda_iobind and runner._sess_euler is not None),
        "warmup": warmup,
        "runs": runs,
        "device_compute_ms": {
            "mean": round(mean_compute, 3),
            "p50": round(float(np.percentile(compute_ms, 50)), 3),
            "p95": round(float(np.percentile(compute_ms, 95)), 3),
            "min": round(float(np.min(compute_ms)), 3),
        },
        "total_e2e_ms": {
            "mean": round(mean_e2e, 3),
            "p50": round(float(np.percentile(e2e_ms, 50)), 3),
            "p95": round(float(np.percentile(e2e_ms, 95)), 3),
            "min": round(float(np.min(e2e_ms)), 3),
        },
        "breakdown_mean_ms": {
            "h2d_ms": round(float(np.mean([r["h2d_ms"] for r in records])), 3),
            "vision_ms": round(float(np.mean([r["vision_ms"] for r in records])), 3),
            "prefix_ms": round(float(np.mean([r["prefix_ms"] for r in records])), 3),
            "expert_10steps_ms": round(float(np.mean([r["expert_10steps_ms"] for r in records])), 3),
            "d2h_ms": round(float(np.mean([r["d2h_ms"] for r in records])), 3),
        },
        "throughput": {
            "chunks_per_sec_e2e": round(1000.0 / max(mean_e2e, 1e-6), 2),
            "control_actions_per_sec_e2e": round((1000.0 / max(mean_e2e, 1e-6)) * CHUNK_SIZE, 1),
        },
        "output_shape": list(last_actions.shape),
        "output_finite": True,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone PyTorch-free SmolVLA ONNX I/O-Binding Runner.")
    parser.add_argument("--onnx-dir", type=Path, required=True, help="Directory with 3 SmolVLA ONNX subgraphs.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--provider", choices=["auto", "cuda", "tensorrt", "cpu"], default="auto")
    parser.add_argument("--benchmark", action="store_true", help="Run synthetic latency/throughput benchmark.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.benchmark:
        runner = SmolVLAOnnxRunner(onnx_dir=args.onnx_dir, device=args.device, provider=args.provider)
        print(f"Loaded SmolVLAOnnxRunner from {args.onnx_dir} on {args.device} ({runner.active_providers})")
        return 0

    summary = run_benchmark(
        onnx_dir=args.onnx_dir,
        device=args.device,
        provider=args.provider,
        warmup=args.warmup,
        runs=args.runs,
    )
    print(json.dumps(summary, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
