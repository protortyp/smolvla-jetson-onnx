#!/usr/bin/env python3
"""Export SmolVLA (`lerobot/smolvla_base`) into 3 static-shape ONNX subgraphs.

Exports:
  1. `smolvla_vision_encoder.onnx`     (SigLIP ViT + 4x4 pixel-shuffle connector + * sqrt(960))
  2. `smolvla_vlm_prefix.onnx`         (embed_tokens + state_proj + 16 VLM layers -> post-RoPE KV cache)
  3. `smolvla_action_expert_step.onnx` (action_in_proj + sinusoidal time + MLP + 16 expert layers + norm + out)

Supports FP32 export, numerically safe mixed-precision FP16 conversion (preserving FP32 on
LayerNorm/RMSNorm, Softmax, RoPE sin/cos, sinusoidal timestep, and I/O boundaries), and
INT8 dynamic weight quantization for CPU deployment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_REPO_ID = "lerobot/smolvla_base"
MODEL_REVISION = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"
BASE_VLM_REPO_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

NUM_CAMERAS = 3
IMAGE_SIZE = 512
PATCH_SIZE = 16
NUM_PATCHES_PER_SIDE = IMAGE_SIZE // PATCH_SIZE  # 32
VISION_SEQ_LEN = NUM_PATCHES_PER_SIDE * NUM_PATCHES_PER_SIDE  # 1024
PIXEL_SHUFFLE_FACTOR = 4
VISION_OUT_TOKENS = VISION_SEQ_LEN // (PIXEL_SHUFFLE_FACTOR**2)  # 64
VISION_HIDDEN = 768
VLM_HIDDEN = 960
EXPERT_HIDDEN = 720
LANG_MAX_LEN = 48
MAX_STATE_DIM = 32
MAX_ACTION_DIM = 32
CHUNK_SIZE = 50
NUM_VLM_LAYERS = 16
NUM_ATTN_HEADS = 15
NUM_KV_HEADS = 5
NUM_KV_GROUPS = NUM_ATTN_HEADS // NUM_KV_HEADS  # 3
HEAD_DIM = 64
PREFIX_LEN = VISION_OUT_TOKENS * NUM_CAMERAS + LANG_MAX_LEN + 1  # 241
ROPE_MAX_WAVELENGTH = 10_000.0
MIN_PERIOD = 0.004
MAX_PERIOD = 4.0
# Finite large negative fill value so all-masked padded query rows produce uniform softmax
# (fill - fill == 0.0) without ever overflowing to -inf -> NaN in FP16 / TensorRT.
ATTN_MASK_FILL = -1.0e9

TRACK_DIR = Path(__file__).resolve().parent
REF_CACHE_PATH = TRACK_DIR / "datasets" / "held_out" / "smolvla_ref_batches.pt"


def _apply_rope_precomputed(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply precomputed RoPE cos/sin [B, L, 1, D/2] to x [B, L, H, D] in float32."""
    d_half = x.shape[-1] // 2
    x_f32 = x.to(torch.float32)
    x1 = x_f32[..., :d_half]
    x2 = x_f32[..., d_half:]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat([y1, y2], dim=-1).to(x.dtype)


def _compute_rope_cos_sin(
    positions: torch.Tensor, timescale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE (cos, sin) of shape [B, L, 1, D/2] matching smolvlm_with_expert.apply_rope."""
    radians = positions.to(torch.float32).unsqueeze(-1) / timescale[None, None, :]
    radians = radians.unsqueeze(-2)
    return torch.cos(radians), torch.sin(radians)


def _eager_gqa_attention(
    attention_mask: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    batch_size: int,
    head_dim: int = HEAD_DIM,
) -> torch.Tensor:
    """Static-shape grouped-query attention matching SmolVLMWithExpertModel.eager_attention_forward.

    Inputs:
      attention_mask: [B, Q_len, K_len] (bool, True = attend)
      query_states:   [B, Q_len, 15, 64]
      key_states:     [B, K_len, 5, 64]
      value_states:   [B, K_len, 5, 64]
    Returns:
      att_output:     [B, Q_len, 960]
    """
    k_len = key_states.shape[1]
    k_exp = key_states.unsqueeze(3).expand(batch_size, k_len, NUM_KV_HEADS, NUM_KV_GROUPS, head_dim)
    k_exp = k_exp.reshape(batch_size, k_len, NUM_ATTN_HEADS, head_dim)

    v_exp = value_states.unsqueeze(3).expand(batch_size, k_len, NUM_KV_HEADS, NUM_KV_GROUPS, head_dim)
    v_exp = v_exp.reshape(batch_size, k_len, NUM_ATTN_HEADS, head_dim)

    q = query_states.to(torch.float32).transpose(1, 2)  # [B, 15, Q_len, 64]
    k = k_exp.to(torch.float32).transpose(1, 2)         # [B, 15, K_len, 64]
    v = v_exp.transpose(1, 2)                           # [B, 15, K_len, 64]

    att_weights = torch.matmul(q, k.transpose(2, 3)) * (head_dim**-0.5)
    fill = torch.tensor(ATTN_MASK_FILL, dtype=torch.float32, device=att_weights.device)
    masked_att_weights = torch.where(attention_mask.unsqueeze(1), att_weights, fill)
    probs = F.softmax(masked_att_weights, dim=-1).to(v.dtype)

    att_output = torch.matmul(probs, v).transpose(1, 2)  # [B, Q_len, 15, 64]
    return att_output.reshape(batch_size, query_states.shape[1], NUM_ATTN_HEADS * head_dim)


class SmolVLAVisionEncoderWrapper(nn.Module):
    """Subgraph 1: Static-shape SigLIP vision encoder + 4x4 pixel-shuffle connector + * sqrt(960).

    Replaces SmolVLMVisionEmbeddings' dynamic `torch.bucketize` + boolean scatter with a
    precomputed exact position embedding buffer `[1, 1024, 768]` for 512x512 inputs, and uses
    clean eager multi-head self-attention for 100% opset-17 / TensorRT 10.x compatibility.
    """

    def __init__(self, vision_model: nn.Module, connector: nn.Module) -> None:
        super().__init__()
        self.patch_embedding = vision_model.embeddings.patch_embedding
        self.encoder_layers = vision_model.encoder.layers
        self.post_layernorm = vision_model.post_layernorm
        self.modality_projection = connector.modality_projection
        self.num_heads = vision_model.config.num_attention_heads
        self.head_dim = vision_model.config.hidden_size // self.num_heads
        self.scale = float(self.head_dim**-0.5)
        self.embed_scale = float(math.sqrt(VLM_HIDDEN))

        # Precompute exact position embeddings for static 512x512 all-valid images
        # using the original embeddings module so bucketize ordering is guaranteed bit-exact.
        with torch.no_grad():
            dummy_pv = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
            dummy_mask = torch.ones(1, NUM_PATCHES_PER_SIDE, NUM_PATCHES_PER_SIDE, dtype=torch.bool)
            patch_emb = self.patch_embedding(dummy_pv).flatten(2).transpose(1, 2)
            full_emb = vision_model.embeddings(pixel_values=dummy_pv, patch_attention_mask=dummy_mask)
            pos_emb = (full_emb - patch_emb).detach().clone()
        self.register_buffer("pos_emb", pos_emb)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        b_cam = pixel_values.shape[0]
        x = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)  # [B_cam, 1024, 768]
        x = x + self.pos_emb

        for layer in self.encoder_layers:
            residual = x
            h = layer.layer_norm1(x)
            q = layer.self_attn.q_proj(h).reshape(b_cam, VISION_SEQ_LEN, self.num_heads, self.head_dim).transpose(1, 2)
            k = layer.self_attn.k_proj(h).reshape(b_cam, VISION_SEQ_LEN, self.num_heads, self.head_dim).transpose(1, 2)
            v = layer.self_attn.v_proj(h).reshape(b_cam, VISION_SEQ_LEN, self.num_heads, self.head_dim).transpose(1, 2)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_probs = F.softmax(attn_weights, dim=-1)
            attn_out = torch.matmul(attn_probs, v).transpose(1, 2).reshape(b_cam, VISION_SEQ_LEN, VISION_HIDDEN)
            x = residual + layer.self_attn.out_proj(attn_out)

            residual = x
            x = residual + layer.mlp(layer.layer_norm2(x))

        x = self.post_layernorm(x)

        # 4x4 pixel shuffle matching SmolVLMConnector.pixel_shuffle(x, scale_factor=4)
        h_grid = NUM_PATCHES_PER_SIDE  # 32
        s = PIXEL_SHUFFLE_FACTOR       # 4
        x = x.reshape(b_cam, h_grid, h_grid // s, VISION_HIDDEN * s)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b_cam, h_grid // s, h_grid // s, VISION_HIDDEN * (s * s))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b_cam, VISION_OUT_TOKENS, VISION_HIDDEN * (s * s))

        x = self.modality_projection(x)
        return x * self.embed_scale


class SmolVLAVLMPrefixWrapper(nn.Module):
    """Subgraph 2: Token embeddings + state_proj + 16 VLM layers -> post-RoPE KV cache.

    Eliminates DynamicCache, text_model.norm, vlm.lm_head (47.3M unused params), and the dead
    attention/MLP tail of layer 15 (9.2M unused params after layer 15 K/V projection).
    """

    def __init__(self, policy_model: nn.Module) -> None:
        super().__init__()
        text_model = policy_model.vlm_with_expert.vlm.model.text_model
        self.embed_tokens = text_model.embed_tokens
        self.state_proj = policy_model.state_proj
        self.layers = text_model.layers[:NUM_VLM_LAYERS]
        self.lang_scale = float(math.sqrt(VLM_HIDDEN))

        # Static prefix attention causal structure: [0]*240 + [1]
        # Image + language tokens (0..239) attend bidirectionally to 0..239;
        # state token (240) attends to all 0..240.
        att_masks = torch.zeros((1, PREFIX_LEN), dtype=torch.int64)
        att_masks[:, -1] = 1
        cumsum = torch.cumsum(att_masks, dim=1)
        prefix_att_const = cumsum[:, None, :] <= cumsum[:, :, None]  # [1, 241, 241] bool
        self.register_buffer("prefix_att_const", prefix_att_const)

        # RoPE timescale matching smolvlm_with_expert.apply_rope (max_wavelength=10_000)
        d_half = HEAD_DIM // 2
        freq_exponents = (2.0 / HEAD_DIM) * torch.arange(d_half, dtype=torch.float32)
        timescale = ROPE_MAX_WAVELENGTH**freq_exponents
        self.register_buffer("rope_timescale", timescale)

    def forward(
        self,
        image_embeddings: torch.Tensor,  # [B, 192, 960]
        img_masks: torch.Tensor,         # [B, 3] bool
        lang_tokens: torch.Tensor,       # [B, 48] int64
        lang_masks: torch.Tensor,        # [B, 48] bool
        state: torch.Tensor,             # [B, 32] float32
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsize = state.shape[0]

        lang_emb = self.embed_tokens(lang_tokens) * self.lang_scale
        state_emb = self.state_proj(state).unsqueeze(1)
        hidden_states = torch.cat([image_embeddings, lang_emb, state_emb], dim=1)  # [B, 241, 960]

        img_pad_masks = img_masks.unsqueeze(-1).expand(bsize, NUM_CAMERAS, VISION_OUT_TOKENS).reshape(
            bsize, NUM_CAMERAS * VISION_OUT_TOKENS
        )
        state_pad_mask = torch.ones((bsize, 1), dtype=torch.bool, device=img_masks.device)
        prefix_pad_masks = torch.cat([img_pad_masks, lang_masks, state_pad_mask], dim=1)  # [B, 241]

        pad_2d = prefix_pad_masks.unsqueeze(1) & prefix_pad_masks.unsqueeze(2)
        prefix_att_2d_masks = self.prefix_att_const & pad_2d

        prefix_position_ids = torch.cumsum(prefix_pad_masks.to(torch.int64), dim=1) - 1
        cos, sin = _compute_rope_cos_sin(prefix_position_ids, self.rope_timescale)

        past_keys_list: list[torch.Tensor] = []
        past_values_list: list[torch.Tensor] = []

        for layer_idx in range(NUM_VLM_LAYERS):
            layer = self.layers[layer_idx]
            normed = layer.input_layernorm(hidden_states)

            k = layer.self_attn.k_proj(normed).reshape(bsize, PREFIX_LEN, NUM_KV_HEADS, HEAD_DIM)
            v = layer.self_attn.v_proj(normed).reshape(bsize, PREFIX_LEN, NUM_KV_HEADS, HEAD_DIM)
            k = _apply_rope_precomputed(k, cos, sin)

            past_keys_list.append(k.transpose(1, 2))    # [B, 5, 241, 64]
            past_values_list.append(v.transpose(1, 2))  # [B, 5, 241, 64]

            # Layer 15's output hidden_states is never consumed (text_model.norm & lm_head unused).
            if layer_idx == NUM_VLM_LAYERS - 1:
                break

            q = layer.self_attn.q_proj(normed).reshape(bsize, PREFIX_LEN, NUM_ATTN_HEADS, HEAD_DIM)
            q = _apply_rope_precomputed(q, cos, sin)

            att_output = _eager_gqa_attention(prefix_att_2d_masks, q, k, v, batch_size=bsize)
            out_emb = layer.self_attn.o_proj(att_output) + hidden_states
            hidden_states = out_emb + layer.mlp(layer.post_attention_layernorm(out_emb))

        past_keys = torch.stack(past_keys_list, dim=0)      # [16, B, 5, 241, 64]
        past_values = torch.stack(past_values_list, dim=0)  # [16, B, 5, 241, 64]
        return past_keys, past_values, prefix_pad_masks


class SmolVLAActionExpertStepWrapper(nn.Module):
    """Subgraph 3: Stateless single Euler flow-matching denoising step for SmolVLA action expert.

    Reads stacked post-RoPE prefix `past_keys` / `past_values` without mutating or cropping cache:
      - Even layers (0, 2, ..., 14): joint self-attention over [prefix_kv; suffix_kv]
      - Odd layers  (1, 3, ..., 15): cross-attention to prefix_kv via expert k_proj/v_proj
    """

    def __init__(self, policy_model: nn.Module) -> None:
        super().__init__()
        self.action_in_proj = policy_model.action_in_proj
        self.action_time_mlp_in = policy_model.action_time_mlp_in
        self.action_time_mlp_out = policy_model.action_time_mlp_out
        self.expert_layers = policy_model.vlm_with_expert.lm_expert.layers[:NUM_VLM_LAYERS]
        self.expert_norm = policy_model.vlm_with_expert.lm_expert.norm
        self.action_out_proj = policy_model.action_out_proj

        # Sinusoidal timestep scaling factors precomputed in float64 for exact fidelity
        fraction = torch.linspace(0.0, 1.0, EXPERT_HIDDEN // 2, dtype=torch.float64)
        period = MIN_PERIOD * (MAX_PERIOD / MIN_PERIOD) ** fraction
        time_scaling_factor = ((1.0 / period) * (2.0 * math.pi)).to(torch.float32)
        self.register_buffer("time_scaling_factor", time_scaling_factor)

        # Causal lower-triangular mask for the 50 action chunk tokens (att_masks = [1]*50)
        suffix_causal = torch.tril(torch.ones((1, CHUNK_SIZE, CHUNK_SIZE), dtype=torch.bool))
        self.register_buffer("suffix_causal_mask", suffix_causal)

        # Constant suffix arange [0..49] and RoPE timescale
        suffix_arange = torch.arange(CHUNK_SIZE, dtype=torch.int64).unsqueeze(0)  # [1, 50]
        self.register_buffer("suffix_arange", suffix_arange)

        d_half = HEAD_DIM // 2
        freq_exponents = (2.0 / HEAD_DIM) * torch.arange(d_half, dtype=torch.float32)
        timescale = ROPE_MAX_WAVELENGTH**freq_exponents
        self.register_buffer("rope_timescale", timescale)

        # Odd cross-attention layers reset expert_position_id to start from 0 -> constant cos/sin
        cos_odd, sin_odd = _compute_rope_cos_sin(suffix_arange, timescale)
        self.register_buffer("cos_odd", cos_odd)
        self.register_buffer("sin_odd", sin_odd)

    def forward(
        self,
        x_t: torch.Tensor,               # [B, 50, 32]
        timestep: torch.Tensor,          # [B]
        past_keys: torch.Tensor,         # [16, B, 5, 241, 64]
        past_values: torch.Tensor,       # [16, B, 5, 241, 64]
        prefix_pad_masks: torch.Tensor,  # [B, 241] bool
    ) -> torch.Tensor:
        bsize = x_t.shape[0]

        # 1. Embed suffix (noisy actions + sinusoidal timestep)
        action_emb = self.action_in_proj(x_t)  # [B, 50, 720]
        sin_input = timestep.to(torch.float32).unsqueeze(-1) * self.time_scaling_factor  # [B, 360]
        time_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1).to(action_emb.dtype)
        time_emb = time_emb.unsqueeze(1).expand(bsize, CHUNK_SIZE, EXPERT_HIDDEN)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)  # [B, 50, 1440]
        hidden_states = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(action_time_emb)))

        # 2. Build attention masks & RoPE tables once for all 16 expert layers
        prefix_pad_2d = prefix_pad_masks.unsqueeze(1).expand(bsize, CHUNK_SIZE, PREFIX_LEN)
        suffix_causal = self.suffix_causal_mask.expand(bsize, CHUNK_SIZE, CHUNK_SIZE)
        full_att_2d_masks = torch.cat([prefix_pad_2d, suffix_causal], dim=2)  # [B, 50, 291]

        prefix_offsets = torch.sum(prefix_pad_masks.to(torch.int64), dim=-1, keepdim=True)  # [B, 1]
        even_position_ids = prefix_offsets + self.suffix_arange  # [B, 50]
        cos_even, sin_even = _compute_rope_cos_sin(even_position_ids, self.rope_timescale)

        # 3. 16 alternating self-attention (even) and cross-attention (odd) expert layers
        for layer_idx in range(NUM_VLM_LAYERS):
            layer = self.expert_layers[layer_idx]
            normed = layer.input_layernorm(hidden_states)
            q = layer.self_attn.q_proj(normed).reshape(bsize, CHUNK_SIZE, NUM_ATTN_HEADS, HEAD_DIM)

            # Prefix KV slice for this layer: [B, 5, 241, 64] -> [B, 241, 5, 64]
            pk = past_keys[layer_idx].transpose(1, 2)
            pv = past_values[layer_idx].transpose(1, 2)

            if layer_idx % 2 == 0:
                # Joint self-attention over [prefix_kv; suffix_kv]
                k_suf = layer.self_attn.k_proj(normed).reshape(bsize, CHUNK_SIZE, NUM_KV_HEADS, HEAD_DIM)
                v_suf = layer.self_attn.v_proj(normed).reshape(bsize, CHUNK_SIZE, NUM_KV_HEADS, HEAD_DIM)
                q = _apply_rope_precomputed(q, cos_even, sin_even)
                k_suf = _apply_rope_precomputed(k_suf, cos_even, sin_even)
                k_full = torch.cat([pk, k_suf], dim=1)
                v_full = torch.cat([pv, v_suf], dim=1)
                att_output = _eager_gqa_attention(full_att_2d_masks, q, k_full, v_full, batch_size=bsize)
            else:
                # Cross-attention to prefix_kv projected through expert k_proj/v_proj (320 -> 320)
                pk_flat = pk.reshape(bsize, PREFIX_LEN, NUM_KV_HEADS * HEAD_DIM)
                pv_flat = pv.reshape(bsize, PREFIX_LEN, NUM_KV_HEADS * HEAD_DIM)
                k_cross = layer.self_attn.k_proj(pk_flat).reshape(bsize, PREFIX_LEN, NUM_KV_HEADS, HEAD_DIM)
                v_cross = layer.self_attn.v_proj(pv_flat).reshape(bsize, PREFIX_LEN, NUM_KV_HEADS, HEAD_DIM)
                q = _apply_rope_precomputed(q, self.cos_odd, self.sin_odd)
                att_output = _eager_gqa_attention(prefix_pad_2d, q, k_cross, v_cross, batch_size=bsize)

            out_emb = layer.self_attn.o_proj(att_output) + hidden_states
            hidden_states = out_emb + layer.mlp(layer.post_attention_layernorm(out_emb))

        hidden_states = self.expert_norm(hidden_states)
        v_t = self.action_out_proj(hidden_states.to(torch.float32))
        return v_t


def _resolve_model_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(MODEL_REPO_ID, revision=MODEL_REVISION, local_files_only=True)
    except Exception:
        local_dir = snapshot_download(MODEL_REPO_ID, revision=MODEL_REVISION, local_files_only=False)
    return Path(local_dir)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _inspect_onnx_model(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(str(path))
    op_counts = Counter(node.op_type for node in model.graph.node)
    total_init_elements = 0
    dtype_counts: Counter[str] = Counter()
    for init in model.graph.initializer:
        elems = 1
        for d in init.dims:
            elems *= int(d)
        total_init_elements += elems
        dtype_counts[onnx.TensorProto.DataType.Name(init.data_type)] += elems

    def _tensor_info(vi: Any) -> dict[str, Any]:
        tt = vi.type.tensor_type
        dims = [
            (d.dim_value if d.dim_value > 0 else d.dim_param)
            for d in tt.shape.dim
        ]
        return {
            "name": vi.name,
            "dtype": onnx.TensorProto.DataType.Name(tt.elem_type),
            "shape": dims,
        }

    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "size_mb": round(path.stat().st_size / (1024.0**2), 2),
        "sha256": _sha256_file(path),
        "opset_version": model.opset_import[0].version if model.opset_import else None,
        "num_nodes": len(model.graph.node),
        "initializer_elements": total_init_elements,
        "initializer_elements_by_dtype": dict(dtype_counts),
        "inputs": [_tensor_info(i) for i in model.graph.input],
        "outputs": [_tensor_info(o) for o in model.graph.output],
        "op_histogram": dict(op_counts.most_common()),
    }


def _verify_pytorch_wrappers(
    vision_w: nn.Module,
    prefix_w: nn.Module,
    expert_w: nn.Module,
    ref_path: Path,
) -> dict[str, float]:
    """Verify the 3 PyTorch wrappers against cached FP32 reference batch 0 before ONNX export."""
    if not ref_path.exists():
        print(f"[warn] Reference cache not found at {ref_path}; skipping pre-export parity check.")
        return {}
    payload = torch.load(ref_path, map_location="cpu", weights_only=True)
    b0 = payload["batches"][0]
    with torch.no_grad():
        img_emb = vision_w(b0["pixel_values"])
        past_k, past_v, pad_m = prefix_w(
            img_emb.reshape(1, NUM_CAMERAS * VISION_OUT_TOKENS, VLM_HIDDEN),
            b0["img_masks"],
            b0["lang_tokens"],
            b0["lang_masks"],
            b0["state"],
        )
        v_t = expert_w(
            b0["x_1"],
            torch.ones((1,), dtype=torch.float32),
            b0["ref_past_keys"],
            b0["ref_past_values"],
            b0["ref_prefix_pad_masks"],
        )
        # 10-step Euler
        x_t = b0["x_1"].clone()
        dt = -0.1
        for step in range(10):
            t_cur = torch.full((1,), 1.0 + step * dt, dtype=torch.float32)
            v_cur = expert_w(x_t, t_cur, past_k, past_v, pad_m)
            x_t = x_t + dt * v_cur

    errs = {
        "vision_max_abs_err": float((img_emb - b0["ref_image_embeddings"]).abs().max().item()),
        "past_keys_max_abs_err": float((past_k - b0["ref_past_keys"]).abs().max().item()),
        "past_values_max_abs_err": float((past_v - b0["ref_past_values"]).abs().max().item()),
        "pad_masks_exact": bool(torch.equal(pad_m, b0["ref_prefix_pad_masks"])),
        "v_t_max_abs_err": float((v_t - b0["ref_v_t"]).abs().max().item()),
        "actions_10step_max_abs_err": float((x_t - b0["ref_actions"]).abs().max().item()),
    }
    print("[pre-export PyTorch wrapper check]", json.dumps(errs, indent=2))
    if not errs["pad_masks_exact"]:
        raise RuntimeError("Pre-export check failed: prefix_pad_masks mismatch!")
    if errs["actions_10step_max_abs_err"] > 1e-4:
        raise RuntimeError(
            f"Pre-export check failed: actions_10step_max_abs_err={errs['actions_10step_max_abs_err']:.3e} > 1e-4"
        )
    return errs


def _write_dir_metadata(
    out_dir: Path,
    precision: str,
    opset: int,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    subgraphs = {}
    total_bytes = 0
    total_init_elements = 0
    for key, fname in [
        ("vision_encoder", "smolvla_vision_encoder.onnx"),
        ("vlm_prefix", "smolvla_vlm_prefix.onnx"),
        ("action_expert_step", "smolvla_action_expert_step.onnx"),
    ]:
        info = _inspect_onnx_model(out_dir / fname)
        subgraphs[key] = info
        total_bytes += info["size_bytes"]
        total_init_elements += info["initializer_elements"]

    config = {
        "model_type": "smolvla_onnx",
        "source_repo_id": MODEL_REPO_ID,
        "source_revision": MODEL_REVISION,
        "base_vlm_repo_id": BASE_VLM_REPO_ID,
        "license": "Apache-2.0",
        "precision": precision,
        "opset_version": opset,
        "static_shapes": {
            "batch_size": 1,
            "num_cameras": NUM_CAMERAS,
            "image_size": IMAGE_SIZE,
            "vision_tokens_per_camera": VISION_OUT_TOKENS,
            "lang_max_len": LANG_MAX_LEN,
            "max_state_dim": MAX_STATE_DIM,
            "max_action_dim": MAX_ACTION_DIM,
            "chunk_size": CHUNK_SIZE,
            "num_euler_steps": 10,
            "num_vlm_layers": NUM_VLM_LAYERS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "prefix_len": PREFIX_LEN,
        },
        "subgraphs": {
            "vision": "smolvla_vision_encoder.onnx",
            "prefix": "smolvla_vlm_prefix.onnx",
            "expert": "smolvla_action_expert_step.onnx",
        },
        "normalization_note": (
            "Inputs/outputs match lerobot/smolvla_base post-preprocessor / pre-postprocessor tensors "
            "(images resized/padded to 512x512 in [-1,1]; state/action MEAN_STD normalized & padded to 32)."
        ),
    }
    if extra_meta:
        config["export_details"] = extra_meta

    manifest = {
        "precision": precision,
        "source_repo_id": MODEL_REPO_ID,
        "source_revision": MODEL_REVISION,
        "opset_version": opset,
        "total_size_bytes": total_bytes,
        "total_size_mb": round(total_bytes / (1024.0**2), 2),
        "total_initializer_elements": total_init_elements,
        "subgraphs": subgraphs,
    }
    if extra_meta:
        manifest["export_details"] = extra_meta

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "onnx_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def export_fp32(out_dir: Path, opset: int = 17, skip_verify: bool = False) -> dict[str, Any]:
    """Load lerobot/smolvla_base in FP32 and export the 3 clean static-shape ONNX subgraphs."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    local_dir = _resolve_model_snapshot()
    print(f"[export_fp32] Loading {MODEL_REPO_ID} @ {MODEL_REVISION[:12]} from {local_dir} in FP32 (CPU)...")
    policy = SmolVLAPolicy.from_pretrained(str(local_dir), local_files_only=True)
    policy.eval()
    policy = policy.to(device="cpu", dtype=torch.float32)

    vision_w = SmolVLAVisionEncoderWrapper(
        policy.model.vlm_with_expert.vlm.model.vision_model,
        policy.model.vlm_with_expert.vlm.model.connector,
    ).eval()
    prefix_w = SmolVLAVLMPrefixWrapper(policy.model).eval()
    expert_w = SmolVLAActionExpertStepWrapper(policy.model).eval()

    pre_errs = {}
    if not skip_verify:
        pre_errs = _verify_pytorch_wrappers(vision_w, prefix_w, expert_w, REF_CACHE_PATH)

    # Dummy static-shape inputs (batch_size=1, 3 cameras)
    dummy_pixel_values = torch.zeros((NUM_CAMERAS, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    dummy_img_embs = torch.zeros((1, NUM_CAMERAS * VISION_OUT_TOKENS, VLM_HIDDEN), dtype=torch.float32)
    dummy_img_masks = torch.ones((1, NUM_CAMERAS), dtype=torch.bool)
    dummy_lang_tokens = torch.ones((1, LANG_MAX_LEN), dtype=torch.int64)
    dummy_lang_masks = torch.ones((1, LANG_MAX_LEN), dtype=torch.bool)
    dummy_state = torch.zeros((1, MAX_STATE_DIM), dtype=torch.float32)
    dummy_x_t = torch.zeros((1, CHUNK_SIZE, MAX_ACTION_DIM), dtype=torch.float32)
    dummy_t = torch.ones((1,), dtype=torch.float32)
    dummy_pk = torch.zeros((NUM_VLM_LAYERS, 1, NUM_KV_HEADS, PREFIX_LEN, HEAD_DIM), dtype=torch.float32)
    dummy_pv = torch.zeros((NUM_VLM_LAYERS, 1, NUM_KV_HEADS, PREFIX_LEN, HEAD_DIM), dtype=torch.float32)
    dummy_pad = torch.ones((1, PREFIX_LEN), dtype=torch.bool)

    export_kwargs: dict[str, Any] = {
        "opset_version": opset,
        "do_constant_folding": True,
        "dynamo": False,
    }

    print("[export_fp32] Exporting smolvla_vision_encoder.onnx ...")
    with torch.no_grad():
        torch.onnx.export(
            vision_w,
            (dummy_pixel_values,),
            str(out_dir / "smolvla_vision_encoder.onnx"),
            input_names=["pixel_values"],
            output_names=["image_embeddings"],
            **export_kwargs,
        )

    print("[export_fp32] Exporting smolvla_vlm_prefix.onnx ...")
    with torch.no_grad():
        torch.onnx.export(
            prefix_w,
            (dummy_img_embs, dummy_img_masks, dummy_lang_tokens, dummy_lang_masks, dummy_state),
            str(out_dir / "smolvla_vlm_prefix.onnx"),
            input_names=["image_embeddings", "img_masks", "lang_tokens", "lang_masks", "state"],
            output_names=["past_keys", "past_values", "prefix_pad_masks"],
            **export_kwargs,
        )

    print("[export_fp32] Exporting smolvla_action_expert_step.onnx ...")
    with torch.no_grad():
        torch.onnx.export(
            expert_w,
            (dummy_x_t, dummy_t, dummy_pk, dummy_pv, dummy_pad),
            str(out_dir / "smolvla_action_expert_step.onnx"),
            input_names=["x_t", "timestep", "past_keys", "past_values", "prefix_pad_masks"],
            output_names=["v_t"],
            **export_kwargs,
        )

    del policy, vision_w, prefix_w, expert_w
    gc.collect()

    elapsed_s = round(time.perf_counter() - t0, 2)
    manifest = _write_dir_metadata(
        out_dir,
        precision="fp32",
        opset=opset,
        extra_meta={
            "exporter": "torch.onnx.export(dynamo=False, do_constant_folding=True)",
            "attn_mask_fill": ATTN_MASK_FILL,
            "pruned_unused_vlm_layer15_tail": True,
            "pruned_unused_lm_head_and_vlm_norm": True,
            "pre_export_pytorch_check": pre_errs,
            "export_seconds": elapsed_s,
        },
    )
    print(f"[export_fp32] Done in {elapsed_s}s | total_size_mb={manifest['total_size_mb']}")
    return manifest


def _topological_sort_graph(model: onnx.ModelProto) -> onnx.ModelProto:
    """Topologically sort ONNX graph nodes (fixes ORT float16 converter appending input Casts)."""
    available: set[str] = set()
    for inp in model.graph.input:
        available.add(inp.name)
    for init in model.graph.initializer:
        available.add(init.name)
    for sp in model.graph.sparse_initializer:
        available.add(sp.values.name)

    remaining = list(model.graph.node)
    sorted_nodes: list[onnx.NodeProto] = []

    while remaining:
        next_remaining: list[onnx.NodeProto] = []
        progressed = False
        for node in remaining:
            if all((inp == "" or inp in available) for inp in node.input):
                sorted_nodes.append(node)
                for out in node.output:
                    if out:
                        available.add(out)
                progressed = True
            else:
                next_remaining.append(node)
        if not progressed:
            raise RuntimeError(
                f"Failed to topologically sort ONNX graph ({len(next_remaining)} nodes unresolved)."
            )
        remaining = next_remaining

    del model.graph.node[:]
    model.graph.node.extend(sorted_nodes)
    return model


def _collapse_fp32_fp16_fp32_casts(model: onnx.ModelProto) -> onnx.ModelProto:
    """Bypass redundant Cast(FP32->FP16) -> Cast(FP16->FP32) round-trips inserted between blocked ops.

    onnxruntime.transformers.float16 wraps each blocked node independently with
    Cast(to FP32) -> Op -> Cast(to FP16). Between adjacent FP32 ops (e.g., RMSNorm's
    Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul where x^2 can reach ~5e6 > 65504),
    an intermediate Cast(to FP16) overflows to +inf and zeroes out visual tokens!
    Bypassing Cast(to FP16)->Cast(to FP32) pairs restores exact FP32 dataflow and removes
    hundreds of unnecessary Cast kernels.
    """
    from onnx import TensorProto

    def _cast_to(node: onnx.NodeProto) -> int | None:
        if node.op_type != "Cast":
            return None
        for attr in node.attribute:
            if attr.name == "to":
                return int(attr.i)
        return None

    producer_by_out: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for out in node.output:
            if out:
                producer_by_out[out] = node

    # Map Cast(to FP32) output -> original FP32 source tensor before Cast(to FP16)
    bypass_map: dict[str, str] = {}
    for node in model.graph.node:
        if _cast_to(node) == TensorProto.FLOAT and len(node.input) == 1:
            prev = producer_by_out.get(node.input[0])
            if prev is not None and _cast_to(prev) == TensorProto.FLOAT16 and len(prev.input) == 1:
                bypass_map[node.output[0]] = prev.input[0]

    # Rewire all consumers
    if bypass_map:
        for node in model.graph.node:
            for idx, inp in enumerate(node.input):
                if inp in bypass_map:
                    node.input[idx] = bypass_map[inp]
        for g_out in model.graph.output:
            if g_out.name in bypass_map:
                g_out.name = bypass_map[g_out.name]

        # Prune dead Cast nodes whose outputs are no longer referenced
        for _ in range(2):
            used: set[str] = {o.name for o in model.graph.output}
            for node in model.graph.node:
                for inp in node.input:
                    if inp:
                        used.add(inp)
            kept = [
                n for n in model.graph.node
                if not (n.op_type == "Cast" and all(o not in used for o in n.output))
            ]
            del model.graph.node[:]
            model.graph.node.extend(kept)

    return model


def convert_fp16(fp32_dir: Path, fp16_dir: Path, opset: int = 17) -> dict[str, Any]:
    """Convert FP32 ONNX subgraphs to numerically safe mixed-precision FP16.

    Preserves FP32 on sensitive ops (LayerNormalization, full RMSNorm blocks, Softmax,
    Sin, Cos, Where, CumSum, ReduceSum) and I/O boundaries (`keep_io_types=True`),
    while executing Conv, MatMul, Gemm, MLP activations, and weights in FP16 Tensor Cores.
    """
    from onnxruntime.transformers import float16 as ort_fp16

    fp16_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    safe_op_block_list = sorted(
        set(ort_fp16.DEFAULT_OP_BLOCK_LIST)
        | {
            "LayerNormalization",
            "Softmax",
            "Sin",
            "Cos",
            "Pow",
            "ReduceMean",
            "Sqrt",
            "Div",
            "Where",
            "CumSum",
            "ReduceSum",
        }
    )

    for fname in [
        "smolvla_vision_encoder.onnx",
        "smolvla_vlm_prefix.onnx",
        "smolvla_action_expert_step.onnx",
    ]:
        src = fp32_dir / fname
        dst = fp16_dir / fname
        print(f"[convert_fp16] Converting {fname} -> mixed FP16 ...")
        model = onnx.load(str(src))
        # Block entire RMSNorm / LayerNorm sub-blocks (including internal Add/Mul) in FP32
        norm_nodes = [
            n.name
            for n in model.graph.node
            if n.name
            and (
                "layernorm" in n.name.lower()
                or "expert_norm" in n.name.lower()
                or "layer_norm" in n.name.lower()
            )
        ]
        model_fp16 = ort_fp16.convert_float_to_float16(
            model,
            min_positive_val=1e-7,
            max_finite_val=65504.0,
            keep_io_types=True,
            disable_shape_infer=False,
            op_block_list=safe_op_block_list,
            node_block_list=norm_nodes,
            force_fp16_initializers=False,
        )
        model_fp16 = _collapse_fp32_fp16_fp32_casts(model_fp16)
        model_fp16 = _topological_sort_graph(model_fp16)
        onnx.save(model_fp16, str(dst))
        del model, model_fp16
        gc.collect()

    elapsed_s = round(time.perf_counter() - t0, 2)
    manifest = _write_dir_metadata(
        fp16_dir,
        precision="fp16",
        opset=opset,
        extra_meta={
            "converter": "onnxruntime.transformers.float16.convert_float_to_float16",
            "keep_io_types": True,
            "fp32_preserved_ops": safe_op_block_list,
            "fp32_preserved_norm_subblocks": True,
            "collapsed_fp32_fp16_fp32_roundtrips": True,
            "convert_seconds": elapsed_s,
        },
    )
    print(f"[convert_fp16] Done in {elapsed_s}s | total_size_mb={manifest['total_size_mb']}")
    return manifest


def convert_int8_weight_only(fp32_dir: Path, int8_dir: Path, opset: int = 17) -> dict[str, Any]:
    """Convert FP32 ONNX subgraphs to CPU-friendly W8A32 weight-only INT8 (MatMulNBits 8-bit).

    Naive dynamic W8A8 (`quantize_dynamic`) degrades SigLIP visual embeddings (Cos ~0.989)
    due to activation outliers. Block-wise symmetric weight-only INT8 (`MatMulNBits`,
    bits=8, block_size=128) keeps activations in FP32 while compressing linear weights to
    8-bit (~539 MB total, 2.8x smaller than FP32) and achieves Actions Cos > 0.999 on CPU.
    """
    import logging
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    int8_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    q_logger = logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer")
    prev_level = q_logger.level
    q_logger.setLevel(logging.WARNING)
    try:
        for fname in [
            "smolvla_vision_encoder.onnx",
            "smolvla_vlm_prefix.onnx",
            "smolvla_action_expert_step.onnx",
        ]:
            src = fp32_dir / fname
            dst = int8_dir / fname
            print(f"[convert_int8] Quantizing {fname} -> W8A32 MatMulNBits (bits=8, block_size=128) ...")
            model = onnx.load(str(src))
            quantizer = MatMulNBitsQuantizer(
                model=model,
                bits=8,
                block_size=128,
                is_symmetric=True,
                accuracy_level=4,
            )
            quantizer.process()
            quantizer.model.save_model_to_file(str(dst))
            del model, quantizer
            gc.collect()
    finally:
        q_logger.setLevel(prev_level)

    elapsed_s = round(time.perf_counter() - t0, 2)
    manifest = _write_dir_metadata(
        int8_dir,
        precision="int8",
        opset=opset,
        extra_meta={
            "converter": "onnxruntime.quantization.matmul_nbits_quantizer.MatMulNBitsQuantizer",
            "bits": 8,
            "block_size": 128,
            "is_symmetric": True,
            "accuracy_level": 4,
            "quantization_scheme": "W8A32 weight-only blockwise INT8 (com.microsoft:MatMulNBits)",
            "convert_seconds": elapsed_s,
        },
    )
    print(f"[convert_int8] Done in {elapsed_s}s | total_size_mb={manifest['total_size_mb']}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SmolVLA to static-shape FP32/FP16/INT8 ONNX.")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=TRACK_DIR / "artifacts",
        help="Root directory where smolvla_onnx_{fp32,fp16,int8} will be written.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17).")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["fp32", "fp16", "int8"],
        default=["fp32", "fp16", "int8"],
        help="Export/conversion variants to build.",
    )
    parser.add_argument("--skip-verify", action="store_true", help="Skip pre-export PyTorch parity check.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fp32_dir = args.artifacts_root / "smolvla_onnx_fp32"
    fp16_dir = args.artifacts_root / "smolvla_onnx_fp16"
    int8_dir = args.artifacts_root / "smolvla_onnx_int8"

    if "fp32" in args.variants:
        export_fp32(fp32_dir, opset=args.opset, skip_verify=args.skip_verify)
    if "fp16" in args.variants:
        if not (fp32_dir / "smolvla_vision_encoder.onnx").exists():
            raise FileNotFoundError(f"FP32 ONNX models not found in {fp32_dir}; run with fp32 first.")
        convert_fp16(fp32_dir, fp16_dir, opset=args.opset)
    if "int8" in args.variants:
        if not (fp32_dir / "smolvla_vision_encoder.onnx").exists():
            raise FileNotFoundError(f"FP32 ONNX models not found in {fp32_dir}; run with fp32 first.")
        convert_int8_weight_only(fp32_dir, int8_dir, opset=args.opset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
