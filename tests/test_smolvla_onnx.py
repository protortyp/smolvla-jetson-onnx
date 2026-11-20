"""CPU unit tests for SmolVLA static ONNX export helpers and PyTorch-free I/O-Binding runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from smolvla_onnx.export_smolvla_onnx import _collapse_fp32_fp16_fp32_casts, convert_fp16
from smolvla_onnx.infer_io_binding import (
    CHUNK_SIZE,
    HEAD_DIM,
    IMAGE_SIZE,
    LANG_MAX_LEN,
    MAX_ACTION_DIM,
    MAX_STATE_DIM,
    NUM_CAMERAS,
    NUM_KV_HEADS,
    NUM_VLM_LAYERS,
    PREFIX_LEN,
    VISION_OUT_TOKENS,
    VLM_HIDDEN,
    SmolVLAOnnxRunner,
    _build_euler_step_onnx_bytes,
)


def _write_dummy_smolvla_subgraphs(onnx_dir: Path) -> None:
    """Create tiny valid FP32 ONNX graphs matching the 3 SmolVLA subgraph I/O signatures."""
    onnx_dir.mkdir(parents=True, exist_ok=True)

    # 1. Vision encoder: pixel_values [3,3,512,512] -> image_embeddings [3,64,960]
    w_vis = helper.make_tensor(
        "vis_zero",
        TensorProto.FLOAT,
        [NUM_CAMERAS, VISION_OUT_TOKENS, VLM_HIDDEN],
        [0.0] * (NUM_CAMERAS * VISION_OUT_TOKENS * VLM_HIDDEN),
    )
    vis_reduce = helper.make_node(
        "ReduceMean", inputs=["pixel_values"], outputs=["vis_mean"], axes=[1, 2, 3], keepdims=True
    )
    vis_shape = helper.make_tensor("vis_shape", TensorProto.INT64, [3], [NUM_CAMERAS, 1, 1])
    vis_reshape = helper.make_node("Reshape", inputs=["vis_mean", "vis_shape"], outputs=["vis_scalar"])
    vis_add = helper.make_node("Add", inputs=["vis_zero", "vis_scalar"], outputs=["image_embeddings"])
    g_vis = helper.make_graph(
        [vis_reduce, vis_reshape, vis_add],
        "dummy_vision",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [NUM_CAMERAS, 3, IMAGE_SIZE, IMAGE_SIZE])],
        [
            helper.make_tensor_value_info(
                "image_embeddings", TensorProto.FLOAT, [NUM_CAMERAS, VISION_OUT_TOKENS, VLM_HIDDEN]
            )
        ],
        initializer=[w_vis, vis_shape],
    )
    onnx.save(
        helper.make_model(g_vis, ir_version=8, opset_imports=[helper.make_opsetid("", 17)]),
        str(onnx_dir / "smolvla_vision_encoder.onnx"),
    )

    # 2. VLM prefix: -> past_keys [16,1,5,241,64], past_values [16,1,5,241,64], prefix_pad_masks [1,241]
    kv_shape = [NUM_VLM_LAYERS, 1, NUM_KV_HEADS, PREFIX_LEN, HEAD_DIM]
    kv_zeros = helper.make_tensor("kv_zero", TensorProto.FLOAT, kv_shape, [0.0] * int(np.prod(kv_shape)))
    pad_ones = helper.make_tensor("pad_ones", TensorProto.BOOL, [1, PREFIX_LEN], [True] * PREFIX_LEN)
    state_sum = helper.make_node("ReduceMean", inputs=["state"], outputs=["st_m"], axes=[1], keepdims=True)
    st_shape = helper.make_tensor("st_shape", TensorProto.INT64, [5], [1, 1, 1, 1, 1])
    st_res = helper.make_node("Reshape", inputs=["st_m", "st_shape"], outputs=["st_bc"])
    pk_node = helper.make_node("Add", inputs=["kv_zero", "st_bc"], outputs=["past_keys"])
    pv_node = helper.make_node("Add", inputs=["kv_zero", "st_bc"], outputs=["past_values"])
    pm_node = helper.make_node("Identity", inputs=["pad_ones"], outputs=["prefix_pad_masks"])
    g_pref = helper.make_graph(
        [state_sum, st_res, pk_node, pv_node, pm_node],
        "dummy_prefix",
        [
            helper.make_tensor_value_info(
                "image_embeddings", TensorProto.FLOAT, [1, NUM_CAMERAS * VISION_OUT_TOKENS, VLM_HIDDEN]
            ),
            helper.make_tensor_value_info("img_masks", TensorProto.BOOL, [1, NUM_CAMERAS]),
            helper.make_tensor_value_info("lang_tokens", TensorProto.INT64, [1, LANG_MAX_LEN]),
            helper.make_tensor_value_info("lang_masks", TensorProto.BOOL, [1, LANG_MAX_LEN]),
            helper.make_tensor_value_info("state", TensorProto.FLOAT, [1, MAX_STATE_DIM]),
        ],
        [
            helper.make_tensor_value_info("past_keys", TensorProto.FLOAT, kv_shape),
            helper.make_tensor_value_info("past_values", TensorProto.FLOAT, kv_shape),
            helper.make_tensor_value_info("prefix_pad_masks", TensorProto.BOOL, [1, PREFIX_LEN]),
        ],
        initializer=[kv_zeros, pad_ones, st_shape],
    )
    onnx.save(
        helper.make_model(g_pref, ir_version=8, opset_imports=[helper.make_opsetid("", 17)]),
        str(onnx_dir / "smolvla_vlm_prefix.onnx"),
    )

    # 3. Action expert step: x_t, timestep, past_keys, past_values, prefix_pad_masks -> v_t [1,50,32]
    scale = helper.make_tensor("v_scale", TensorProto.FLOAT, [1], [0.5])
    mul_node = helper.make_node("Mul", inputs=["x_t", "v_scale"], outputs=["v_t"])
    g_exp = helper.make_graph(
        [mul_node],
        "dummy_expert",
        [
            helper.make_tensor_value_info("x_t", TensorProto.FLOAT, [1, CHUNK_SIZE, MAX_ACTION_DIM]),
            helper.make_tensor_value_info("timestep", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("past_keys", TensorProto.FLOAT, kv_shape),
            helper.make_tensor_value_info("past_values", TensorProto.FLOAT, kv_shape),
            helper.make_tensor_value_info("prefix_pad_masks", TensorProto.BOOL, [1, PREFIX_LEN]),
        ],
        [helper.make_tensor_value_info("v_t", TensorProto.FLOAT, [1, CHUNK_SIZE, MAX_ACTION_DIM])],
        initializer=[scale],
    )
    onnx.save(
        helper.make_model(g_exp, ir_version=8, opset_imports=[helper.make_opsetid("", 17)]),
        str(onnx_dir / "smolvla_action_expert_step.onnx"),
    )


def test_euler_micro_graph_and_cast_collapse() -> None:
    raw_bytes = _build_euler_step_onnx_bytes(np.dtype(np.float16), np.dtype(np.float32))
    assert raw_bytes is not None
    model = onnx.load_from_string(raw_bytes)
    onnx.checker.check_model(model)
    assert [n.op_type for n in model.graph.node] == ["Cast", "Mul", "Add"]

    c1 = helper.make_node("Cast", inputs=["x"], outputs=["x16"], to=TensorProto.FLOAT16)
    c2 = helper.make_node("Cast", inputs=["x16"], outputs=["x32"], to=TensorProto.FLOAT)
    id_n = helper.make_node("Identity", inputs=["x32"], outputs=["y"])
    g = helper.make_graph(
        [c1, c2, id_n],
        "cast_collapse",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
    )
    collapsed = _collapse_fp32_fp16_fp32_casts(
        helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    )
    assert [n.op_type for n in collapsed.graph.node] == ["Identity"]
    assert collapsed.graph.node[0].input[0] == "x"


def test_smolvla_onnx_runner_cpu_and_fp16_conversion(tmp_path: Path) -> None:
    fp32_dir = tmp_path / "fp32"
    fp16_dir = tmp_path / "fp16"
    _write_dummy_smolvla_subgraphs(fp32_dir)

    manifest = convert_fp16(fp32_dir, fp16_dir)
    assert manifest["precision"] == "fp16"
    assert (fp16_dir / "config.json").is_file()

    runner = SmolVLAOnnxRunner(fp32_dir, device="cpu")
    pixel_values = np.zeros((NUM_CAMERAS, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    img_masks = np.ones((1, NUM_CAMERAS), dtype=np.bool_)
    lang_tokens = np.ones((1, LANG_MAX_LEN), dtype=np.int64)
    lang_masks = np.ones((1, LANG_MAX_LEN), dtype=np.bool_)
    state = np.zeros((1, MAX_STATE_DIM), dtype=np.float32)
    noise = np.ones((1, CHUNK_SIZE, MAX_ACTION_DIM), dtype=np.float32)

    actions, timings = runner.predict_action_chunk(
        pixel_values, img_masks, lang_tokens, lang_masks, state, noise=noise, return_timings=True
    )
    assert actions.shape == (1, CHUNK_SIZE, MAX_ACTION_DIM)
    assert actions.dtype == np.float32
    assert np.all(np.isfinite(actions))
    np.testing.assert_allclose(actions, np.full_like(actions, 0.95**10), rtol=1e-4, atol=1e-5)
    assert timings["total_e2e_ms"] > 0.0
