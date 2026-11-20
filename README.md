# SmolVLA Jetson ONNX (`protortyp/smolvla-jetson-onnx`)

Static ONNX export, mixed FP16 / W8A32 INT8, and PyTorch-free zero-copy CUDA I/O-Binding runner for [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base/tree/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6) (`Apache-2.0`). Weights (`fp16/`, `int8/`, `fp32/`): [`christianengel/smolvla-jetson-onnx`](https://huggingface.co/christianengel/smolvla-jetson-onnx).

RTX 3090 (`8` batches, `B=1`, `3x512x512`, `10` steps): FP16 `SmolVLAOnnxRunner` hits **78.17 ms** (**639.6 actions/s**; **3.11x** vs PT FP32 CUDA, **2.55x** vs BF16 CUDA), **-3.12–3.86 GB** peak RSS, `0.999983` cosine (BF16: `0.999975`).

## Measured benchmarks

| Backend / Variant | Mean (ms) | p50 (ms) | p95 (ms) | Chunks/s | Actions/s | Act Cosine (vs FP32) | Peak RSS (MB) | Disk (MB) | Record |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `pytorch_fp32_cuda` | 243.05 | 244.27 | 251.23 | 4.11 | 205.7 | 1.000000 | 4848.49 | 864.71 | [baseline](results/baseline_pytorch.json) |
| `pytorch_bf16_cuda` | 199.41 | 200.16 | 204.54 | 5.02 | 250.8 | 0.999975 | 5590.92 | 864.71 | [baseline](results/baseline_pytorch.json) |
| `pytorch_fp32_cpu` | 1979.85 | 1981.06 | 2021.18 | 0.51 | 25.3 | 1.000000 | 5947.50 | 864.71 | [baseline](results/baseline_pytorch.json) |
| `onnx_fp32_cuda_iobind` | 132.92 | 125.71 | 155.82 | 7.52 | 376.2 | 1.000000 | 1655.88 | 1501.61 | [fp32 iobind](results/onnx_fp32_cuda_iobind.json) |
| `infer_iobind_fp32_cuda` | 130.78 | 130.45 | 132.77 | 7.65 | 382.3 | 1.000000 | — | 1501.61 | [runner fp32](results/infer_iobind_fp32_cuda.json) |
| `onnx_fp16_cuda_iobind` | 79.85 | 80.06 | 81.86 | 12.52 | 626.2 | 0.999983 | 1730.90 | 751.57 | [fp16 iobind](results/onnx_fp16_cuda_iobind.json) |
| **`infer_iobind_fp16_cuda`** | **78.17** | **78.22** | **79.73** | **12.79** | **639.6** | **0.999983** | **1730.90** | **751.57** | [runner fp16](results/infer_iobind_fp16_cuda.json) |
| `onnx_int8_cuda_iobind` (W8A32) | 102.79 | 102.58 | 105.37 | 9.73 | 486.4 | 0.999727 | 1513.15 | 538.52 | [int8 iobind](results/onnx_int8_cuda_iobind.json) |
| `onnx_int8_cpu` (W8A32) | 2689.45 | 3013.86 | 3121.58 | 0.37 | 18.6 | 0.999268 | 2126.59 | 538.52 | [int8 cpu](results/onnx_int8_cpu.json) |
| `onnx_fp32_cpu` | 3790.80 | 4280.23 | 4382.89 | 0.26 | 13.2 | 1.000000 | 3184.72 | 1501.61 | [fp32 cpu](results/onnx_fp32_cpu.json) |

Docs: [benchmarks](docs/BENCHMARK.md) | [TensorRT](docs/JETSON_TENSORRT.md) | [reproduce](docs/REPRODUCE.md).

## Quick start

```bash
uv pip install -e ".[cuda]"   # or ".[cpu]" for CPU-only
```

Fetch [`config.json`](https://huggingface.co/christianengel/smolvla-jetson-onnx/blob/main/config.json) + `fp16/` from `christianengel/smolvla-jetson-onnx`:

```python
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from smolvla_onnx import SmolVLAOnnxRunner

repo_id = "christianengel/smolvla-jetson-onnx"
hf_hub_download(repo_id=repo_id, filename="config.json", local_dir="./artifacts/smolvla_hf")
snapshot_download(
    repo_id=repo_id,
    allow_patterns=["config.json", "fp16/*"],
    local_dir="./artifacts/smolvla_hf",
)

runner = SmolVLAOnnxRunner("./artifacts/smolvla_hf/fp16", device="cuda")
actions, timings = runner.predict_action_chunk(
    pixel_values=np.zeros((3, 3, 512, 512), dtype=np.float32),
    img_masks=np.ones((1, 3), dtype=np.bool_),
    lang_tokens=np.ones((1, 48), dtype=np.int64),
    lang_masks=np.ones((1, 48), dtype=np.bool_),
    state=np.zeros((1, 32), dtype=np.float32),
    return_timings=True,
)
print(actions.shape, timings)  # (1, 50, 32)
```

## Layout

- [`smolvla_onnx/export_smolvla_onnx.py`](smolvla_onnx/export_smolvla_onnx.py): Export `lerobot/smolvla_base` -> 3 opset-17 subgraphs (`fp32`, `fp16`, W8A32 `int8`).
- [`smolvla_onnx/infer_io_binding.py`](smolvla_onnx/infer_io_binding.py): PyTorch-free `SmolVLAOnnxRunner` (CUDA `OrtValue` Euler loop).
- [`smolvla_onnx/eval_locked.py`](smolvla_onnx/eval_locked.py): Parity, latency, RSS, NVML VRAM harness.
- [`scripts/build_tensorrt_jetson.sh`](scripts/build_tensorrt_jetson.sh): TRT 10.x `trtexec` builder (`orin-nano-8gb`, `orin-nx-16gb` JetPack 6.2; `thor` 7.2).
- [`results/`](results/baseline_pytorch.json): JSON benchmark records.
- [`tests/test_smolvla_onnx.py`](tests/test_smolvla_onnx.py): CPU unit tests.
- [`LICENSE`](LICENSE) & [`NOTICE`](NOTICE): Apache-2.0 (`lerobot/smolvla_base`, `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`).
