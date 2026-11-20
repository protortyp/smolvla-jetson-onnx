# Benchmark report

## Measurement conditions

Ryzen 9 5950X (16C/32T) + RTX 3090 (shared host); `lerobot/smolvla_base@5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`. [`eval_locked.py`](../smolvla_onnx/eval_locked.py): `8` batches (`seeds=[11, 23, 37, 41, 53, 111, 123, 137]`, `B=1`) vs PyTorch FP32 (`allow_tf32=False`, `NVIDIA_TF32_OVERRIDE=0`).

- GPU: `5` warmup, `20` timed (`10` Euler steps).
- CPU: `1` warmup, `3` timed.
- `infer_io_binding.py`: `5` warmup, `20` timed GPU; persistent `OrtValue` + 2-node Euler micro-graph (`x_next = x_t + dt * v_t`).

## Full runtime and memory breakdown

| Backend | Mean (ms) | p50 (ms) | p95 (ms) | H2D (ms) | Vision (ms) | Prefix (ms) | Expert 10x (ms) | D2H (ms) | Chunks/s | Act/s | Peak GPU (MB) | Peak RSS (MB) | Est. Jetson Total (MB) | Disk (MB) | Record |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `pytorch_fp32_cuda` | 243.054 | 244.271 | 251.225 | — | 80.614 | 18.856 | 143.583 | — | 4.114 | 205.7 | 1950.52 alloc / 2388 NVML | 4848.49 | — | 864.71 | [baseline](../results/baseline_pytorch.json) |
| `pytorch_bf16_cuda` | 199.405 | 200.157 | 204.536 | — | 23.128 | 17.056 | 159.220 | — | 5.015 | 250.75 | 1034.33 alloc / 1412 NVML | 5590.92 | — | 864.71 | [baseline](../results/baseline_pytorch.json) |
| `pytorch_fp32_cpu` | 1979.846 | 1981.063 | 2021.177 | — | 1207.600 | 157.969 | 614.277 | — | 0.505 | 25.25 | 0.0 | 5947.50 | — | 864.71 | [baseline](../results/baseline_pytorch.json) |
| `onnx_fp32_cuda` | 146.435 | 135.485 | 189.233 | — | 64.287 | 10.870 | 71.278 | — | 6.829 | 341.45 | 2882.0 | 1638.61 | 2734.11 | 1501.61 | [fp32 cuda](../results/onnx_fp32_cuda.json) |
| `onnx_fp32_cuda_iobind` | 132.922 | 125.714 | 155.817 | — | 62.666 | 9.579 | 60.677 | — | 7.523 | 376.15 | 3138.0 | 1655.88 | 2798.11 | 1501.61 | [fp32 iobind](../results/onnx_fp32_cuda_iobind.json) |
| `infer_iobind_fp32_cuda` | 130.779 | 130.446 | 132.771 | 1.101 | 65.628 | 9.511 | 54.491 | 0.047 | 7.65 | 382.3 | — | — | — | 1501.61 | [runner fp32](../results/infer_iobind_fp32_cuda.json) |
| `onnx_fp16_cuda` | 85.983 | 86.088 | 89.557 | — | 35.109 | 5.259 | 45.614 | — | 11.630 | 581.5 | 2126.0 | 1698.50 | 1795.07 | 751.57 | [fp16 cuda](../results/onnx_fp16_cuda.json) |
| `onnx_fp16_cuda_iobind` | 79.853 | 80.055 | 81.855 | — | 36.375 | 4.221 | 39.258 | — | 12.523 | 626.15 | 2344.0 | 1730.90 | 1849.57 | 751.57 | [fp16 iobind](../results/onnx_fp16_cuda_iobind.json) |
| `infer_iobind_fp16_cuda` | 78.168 | 78.221 | 79.731 | 1.024 | 35.634 | 4.098 | 37.364 | 0.047 | 12.79 | 639.6 | — | — | — | 751.57 | [runner fp16](../results/infer_iobind_fp16_cuda.json) |
| `onnx_int8_cuda_iobind` | 102.792 | 102.579 | 105.371 | — | 45.247 | 6.868 | 50.677 | — | 9.728 | 486.4 | 2334.0 | 1513.15 | 1634.02 | 538.52 | [int8 iobind](../results/onnx_int8_cuda_iobind.json) |
| `onnx_int8_cpu` | 2689.451 | 3013.857 | 3121.575 | — | 1736.761 | 540.724 | 411.966 | — | 0.372 | 18.6 | 0.0 | 2126.59 | 1408.67 | 538.52 | [int8 cpu](../results/onnx_int8_cpu.json) |
| `onnx_fp32_cpu` | 3790.801 | 4280.226 | 4382.887 | — | 2444.404 | 337.221 | 1009.176 | — | 0.264 | 13.2 | 0.0 | 3184.72 | 2636.22 | 1501.61 | [fp32 cpu](../results/onnx_fp32_cpu.json) |
| `onnx_int8_dynamic_w8a8_cpu` | 3697.407 | 3612.029 | 3902.379 | — | 2382.910 | 357.214 | 957.283 | — | 0.270 | 13.5 | 0.0 | 2112.51 | 1384.37 | 517.73 | [w8a8 fail](../results/onnx_int8_dynamic_w8a8_cpu.json) |

## Per-subgraph numerical parity

`prefix_pad_masks_exact_match=true`. Gates: FP32 `actions_max_abs_err < 1e-4` & `actions_cosine_sim >= 0.99999`; FP16 `>= 0.999`; INT8 `>= 0.995`.

| Variant | Output Tensor | MaxAbsErr | Rel MaxAbsErr | MAE | RMSE | Mean Cosine | Min Cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pytorch_bf16_cuda` | `vision_image_embeddings` | 22.510 | 1.041e-02 | 1.080 | 1.504 | 0.999941 | 0.999937 |
| `pytorch_bf16_cuda` | `prefix_past_keys` | 0.3675 | 2.021e-02 | 1.013e-02 | 1.424e-02 | 0.999967 | 0.999964 |
| `pytorch_bf16_cuda` | `prefix_past_values` | 0.1206 | 2.369e-02 | 6.411e-03 | 9.228e-03 | 0.999910 | 0.999904 |
| `pytorch_bf16_cuda` | `expert_single_step_v_t` | 4.416e-02 | 1.118e-02 | 4.131e-03 | 5.247e-03 | 0.999987 | 0.999983 |
| `pytorch_bf16_cuda` | `actions_10step` | 4.519e-02 | 1.503e-02 | 1.110e-03 | 2.138e-03 | 0.999975 | 0.999935 |
| `onnx_fp32_cuda_iobind` | `vision_image_embeddings` | 5.249e-03 | 2.352e-06 | 2.249e-04 | 3.123e-04 | 1.000000 | 1.000000 |
| `onnx_fp32_cuda_iobind` | `prefix_past_keys` | 8.011e-05 | 4.406e-06 | 1.512e-06 | 2.235e-06 | 1.000000 | 1.000000 |
| `onnx_fp32_cuda_iobind` | `prefix_past_values` | 3.517e-05 | 6.907e-06 | 1.038e-06 | 1.573e-06 | 1.000000 | 1.000000 |
| `onnx_fp32_cuda_iobind` | `expert_single_step_v_t` | 1.907e-06 | 4.899e-07 | 2.887e-07 | 3.724e-07 | 1.000000 | 1.000000 |
| `onnx_fp32_cuda_iobind` | `actions_10step` | 3.742e-06 | 2.273e-06 | 1.077e-07 | 2.388e-07 | 1.000000 | 1.000000 |
| `onnx_fp16_cuda_iobind` | `vision_image_embeddings` | 3.671 | 1.626e-03 | 0.1723 | 0.2406 | 0.999998 | 0.999998 |
| `onnx_fp16_cuda_iobind` | `prefix_past_keys` | 0.1878 | 1.033e-02 | 3.298e-03 | 7.593e-03 | 0.999991 | 0.999990 |
| `onnx_fp16_cuda_iobind` | `prefix_past_values` | 2.818e-02 | 6.034e-03 | 9.591e-04 | 1.405e-03 | 0.999998 | 0.999998 |
| `onnx_fp16_cuda_iobind` | `expert_single_step_v_t` | 9.778e-03 | 2.690e-03 | 1.363e-03 | 1.737e-03 | 0.999999 | 0.999998 |
| `onnx_fp16_cuda_iobind` | `actions_10step` | 4.392e-02 | 1.622e-02 | 5.324e-04 | 1.532e-03 | 0.999983 | 0.999954 |
| `onnx_int8_cuda_iobind` | `vision_image_embeddings` | 36.107 | 1.617e-02 | 2.237 | 3.037 | 0.999763 | 0.999746 |
| `onnx_int8_cuda_iobind` | `prefix_past_keys` | 0.3328 | 1.830e-02 | 1.645e-02 | 2.269e-02 | 0.999917 | 0.999909 |
| `onnx_int8_cuda_iobind` | `prefix_past_values` | 0.2128 | 4.063e-02 | 1.154e-02 | 1.620e-02 | 0.999690 | 0.999665 |
| `onnx_int8_cuda_iobind` | `expert_single_step_v_t` | 1.934e-02 | 4.896e-03 | 4.291e-03 | 5.449e-03 | 0.999986 | 0.999984 |
| `onnx_int8_cuda_iobind` | `actions_10step` | 7.411e-02 | 3.735e-02 | 2.875e-03 | 6.080e-03 | 0.999727 | 0.999306 |
| `onnx_int8_cpu` | `vision_image_embeddings` | 75.481 | 3.382e-02 | 4.071 | 5.545 | 0.999198 | 0.999084 |
| `onnx_int8_cpu` | `prefix_past_keys` | 0.8958 | 4.926e-02 | 3.155e-02 | 4.340e-02 | 0.999695 | 0.999677 |
| `onnx_int8_cpu` | `prefix_past_values` | 0.4718 | 8.621e-02 | 2.217e-02 | 3.094e-02 | 0.998768 | 0.998709 |
| `onnx_int8_cpu` | `expert_single_step_v_t` | 6.408e-02 | 1.597e-02 | 8.983e-03 | 1.133e-02 | 0.999937 | 0.999931 |
| `onnx_int8_cpu` | `actions_10step` | 0.1210 | 6.161e-02 | 5.059e-03 | 9.588e-03 | 0.999268 | 0.997738 |
| `onnx_int8_dynamic_w8a8_cpu` (fail) | `vision_image_embeddings` | 229.301 | 1.027e-01 | 14.716 | 20.105 | 0.989682 | 0.988959 |
| `onnx_int8_dynamic_w8a8_cpu` (fail) | `prefix_past_keys` | 4.003 | 2.203e-01 | 0.1134 | 0.1571 | 0.996002 | 0.995607 |
| `onnx_int8_dynamic_w8a8_cpu` (fail) | `prefix_past_values` | 1.684 | 3.307e-01 | 7.945e-02 | 0.1121 | 0.985192 | 0.984160 |
| `onnx_int8_dynamic_w8a8_cpu` (fail) | `expert_single_step_v_t` | 7.461e-02 | 1.992e-02 | 1.252e-02 | 1.586e-02 | 0.999877 | 0.999861 |
| `onnx_int8_dynamic_w8a8_cpu` (fail) | `actions_10step` | 0.5315 | 2.368e-01 | 1.733e-02 | 4.185e-02 | 0.987607 | 0.955954 |

`vision_image_embeddings`: SigLIP connector scales by `sqrt(960) (~30.98)`, pushing activations to `~2,300`; `rel_max_abs_err` stays `2.35e-06` (FP32) and `1.63e-03` (FP16).

## Diagnostic notes

### RMSNorm FP16 overflow and cast-roundtrip collapse

`convert_float_to_float16` wraps blocked ops individually (`Cast(FP32)->Op->Cast(FP16)`). Inside SmolVLM RMSNorm (`Pow->ReduceMean->Add->Sqrt->Div->Mul`), `sqrt(960)`-scaled squares hit `~5e6 > 65504`, overflowing FP16 casts to `+inf` (`0`/`NaN`).

[`export_smolvla_onnx.py`](../smolvla_onnx/export_smolvla_onnx.py):
1. Blocks `LayerNormalization`, `Softmax`, `Sin`, `Cos`, `Pow`, `ReduceMean`, `Sqrt`, `Div`, `Where`, `CumSum`, `ReduceSum` plus `*layernorm*`, `*layer_norm*`, `*expert_norm*` (`node_block_list`).
2. Runs `_collapse_fp32_fp16_fp32_casts` on `Cast(FP32->FP16)->Cast(FP16->FP32)` pairs, keeping RMSNorm in FP32.

### W8A8 vs W8A32 `MatMulNBits`

Dynamic W8A8 (`quantize_dynamic`, `517.73 MB`, `0.270` chunks/s CPU) fails `>= 0.995` cosine (`0.987607` mean, `0.955954` min; [record](../results/onnx_int8_dynamic_w8a8_cpu.json)) from SigLIP outliers (`MaxAbsErr=229.30`, `Cos=0.989682`).

Weight-only INT8 (`MatMulNBitsQuantizer`, `bits=8`, `block_size=128`, `accuracy_level=4`) keeps FP32 activations at `538.52 MB` (`2.79x` smaller than FP32), passing on CUDA (`0.999727`, `102.79 ms`) and CPU (`0.999268`, `2689.45 ms`, `-3,820.91 MB` RSS vs PT FP32 CPU).
