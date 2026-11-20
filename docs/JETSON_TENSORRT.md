# Jetson TensorRT 10.x Deployment Guide

[`scripts/build_tensorrt_jetson.sh`](../scripts/build_tensorrt_jetson.sh) compiles the 3 static SmolVLA subgraphs into TensorRT `.engine` files via `trtexec`.

## Hardware profiles

| Profile (`--profile`) | Target Hardware | JetPack / TRT | Workspace Pool | Builder Opt Level | Precision Mode |
| --- | --- | --- | ---: | ---: | --- |
| `orin-nano-8gb` | Jetson Orin Nano 8 GB (SM 8.7) | JetPack 6.2 / TRT 10.3+ | 1024M | 3 | FP16 + FP32 constraints |
| `orin-nx-16gb` | Jetson Orin NX 16 GB (SM 8.7) | JetPack 6.2 / TRT 10.3+ | 2048M | 4 | FP16 + FP32 constraints |
| `thor` | Jetson Thor (Blackwell) | JetPack 7.2 / TRT 10.x+ | 4096M | 5 | FP16 + FP32 constraints |

## Precision constraints

Flags preventing RMSNorm overflow and attention/RoPE drift:

```text
--fp16
--precisionConstraints=prefer
--layerPrecisions=*LayerNormalization*:fp32,*layernorm*:fp32,*expert_norm*:fp32,*Softmax*:fp32,*Sin*:fp32,*Cos*:fp32,*Where*:fp32,*CumSum*:fp32
```

`--strict-fp32-norms` switches `prefer` to `obey`. `fp32/` (`1,501.61 MB`) lets TRT quantize weights directly under FP32 constraints; `fp16/` (`751.57 MB`) works when disk/bandwidth is tight.

## Jetson Orin Nano 8 GB memory budget

Orin Nano 8 GB shares `8,192 MB` LPDDR5 (OS `~1.5 GB`, camera/ROS2 `~0.5 GB`, workspace, weights). [`eval_locked.py`](../smolvla_onnx/eval_locked.py) enforces `jetson_safe_limit_mb = 6144.0 MB` and `onnx_disk_size_mb <= 2200.0 MB`:

`est_jetson_total_mb = onnx_disk_size_mb + 0.25 * mem_delta_mb + 512.0`

| Variant | Disk Size (MB) | Est. Jetson Total (MB) | Safe Limit (MB) | Status |
| --- | ---: | ---: | ---: | --- |
| `fp16` (CUDA I/O-Bind) | 751.57 | 1849.57 | 6144.0 | Pass |
| `int8` (CUDA I/O-Bind) | 538.52 | 1634.02 | 6144.0 | Pass |
| `int8` (CPU) | 538.52 | 1408.67 | 6144.0 | Pass |
| `fp32` (CUDA I/O-Bind) | 1501.61 | 2798.11 | 6144.0 | Pass |

`--memPoolSize=workspace:1024M` in `orin-nano-8gb` prevents 8 GB builder OOMs.

## Build commands

Fetch [`config.json`](https://huggingface.co/christianengel/smolvla-jetson-onnx/blob/main/config.json) + subgraphs from `christianengel/smolvla-jetson-onnx`, then build:

```bash
hf download christianengel/smolvla-jetson-onnx \
  --include "config.json" "fp16/*" \
  --local-dir ./artifacts/smolvla_hf

sudo nvpmodel -m 0 && sudo jetson_clocks

./scripts/build_tensorrt_jetson.sh \
  --onnx-dir ./artifacts/smolvla_hf/fp16 \
  --out-dir ./artifacts/smolvla_trt_orin_nano \
  --profile orin-nano-8gb
```

Dry-run (non-Jetson):

```bash
./scripts/build_tensorrt_jetson.sh --profile orin-nano-8gb --dry-run
```

> **Verification scope**: `scripts/build_tensorrt_jetson.sh` verified via `--dry-run` only on x86_64 (no `libnvinfer.so`/`trtexec` or Jetson hardware); `.engine` latencies unmeasured.
