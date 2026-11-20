# Reproduction Guide

## 1. Install

Runtime (PyTorch-free, consumes `christianengel/smolvla-jetson-onnx`):

```bash
uv pip install -e ".[cuda]"    # or ".[cpu]"
```

Export & eval (`torch`, `lerobot`, `psutil`, `pytest`):

```bash
uv pip install -e ".[cuda,export]" --group dev
```

## 2. Unit tests

CPU tests for `SmolVLAOnnxRunner`, Euler micro-graph, cast collapse, and `convert_fp16`:

```bash
pytest -q
```

## 3. Export subgraphs

Exports `lerobot/smolvla_base@5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` on CPU to `./artifacts/smolvla_onnx_{fp32,fp16,int8}`:

```bash
python -m smolvla_onnx.export_smolvla_onnx \
  --artifacts-root ./artifacts \
  --variants fp32 fp16 int8
```

Add `--skip-verify` without a `smolvla_ref_batches.pt` cache.

## 4. Locked evaluation

Generate baselines and `datasets/held_out/smolvla_ref_batches.pt`:

```bash
NVIDIA_TF32_OVERRIDE=0 python -m smolvla_onnx.eval_locked --baseline-only
```

Evaluate variants:

```bash
# Strict IEEE FP32 CUDA + I/O-Binding
NVIDIA_TF32_OVERRIDE=0 python -m smolvla_onnx.eval_locked \
  --onnx-dir ./artifacts/smolvla_onnx_fp32 \
  --precision fp32 --device cuda --io-binding \
  --output-json ./results/onnx_fp32_cuda_iobind.json

# Mixed FP16 CUDA + I/O-Binding (recommended)
python -m smolvla_onnx.eval_locked \
  --onnx-dir ./artifacts/smolvla_onnx_fp16 \
  --precision fp16 --device cuda --io-binding \
  --output-json ./results/onnx_fp16_cuda_iobind.json

# W8A32 INT8 CUDA + I/O-Binding and CPU
python -m smolvla_onnx.eval_locked \
  --onnx-dir ./artifacts/smolvla_onnx_int8 \
  --precision int8 --device cuda --io-binding \
  --output-json ./results/onnx_int8_cuda_iobind.json

python -m smolvla_onnx.eval_locked \
  --onnx-dir ./artifacts/smolvla_onnx_int8 \
  --precision int8 --device cpu \
  --output-json ./results/onnx_int8_cpu.json
```

## 5. Standalone runner benchmark

```bash
python -m smolvla_onnx.infer_io_binding \
  --onnx-dir ./artifacts/smolvla_onnx_fp16 \
  --device cuda --benchmark --warmup 5 --runs 20 \
  --output-json ./results/infer_iobind_fp16_cuda.json
```
