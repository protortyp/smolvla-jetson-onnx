"""Static-shape ONNX export and PyTorch-free I/O-Binding inference for SmolVLA."""

from smolvla_onnx.infer_io_binding import (
    CHUNK_SIZE,
    HEAD_DIM,
    IMAGE_SIZE,
    LANG_MAX_LEN,
    MAX_ACTION_DIM,
    MAX_STATE_DIM,
    NUM_CAMERAS,
    NUM_EULER_STEPS,
    NUM_KV_HEADS,
    NUM_VLM_LAYERS,
    PREFIX_LEN,
    VISION_OUT_TOKENS,
    VLM_HIDDEN,
    SmolVLAOnnxRunner,
)

__all__ = [
    "CHUNK_SIZE",
    "HEAD_DIM",
    "IMAGE_SIZE",
    "LANG_MAX_LEN",
    "MAX_ACTION_DIM",
    "MAX_STATE_DIM",
    "NUM_CAMERAS",
    "NUM_EULER_STEPS",
    "NUM_KV_HEADS",
    "NUM_VLM_LAYERS",
    "PREFIX_LEN",
    "SmolVLAOnnxRunner",
    "VISION_OUT_TOKENS",
    "VLM_HIDDEN",
]

__version__ = "0.1.0"
