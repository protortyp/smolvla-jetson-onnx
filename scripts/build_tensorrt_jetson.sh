#!/usr/bin/env bash
# ==============================================================================
# Deterministic TensorRT 10.x Engine Build Script for SmolVLA ONNX Subgraphs
# Target: NVIDIA Jetson Orin Nano 8 GB / Orin NX 16 GB (JetPack 6.2) &
#         NVIDIA Jetson Thor (JetPack 7.2)
#
# Builds 3 static-shape TensorRT engines from the FP32 (or mixed FP16) ONNX
# subgraphs exported by `export_smolvla_onnx.py`:
#   1. smolvla_vision_encoder.engine       (pixel_values: 3x3x512x512)
#   2. smolvla_vlm_prefix.engine           (prefix_len: 241, 16 VLM layers)
#   3. smolvla_action_expert_step.engine   (chunk_size: 50, 16 expert layers)
#
# Recommended before benchmarking on Jetson:
#   sudo nvpmodel -m 0 && sudo jetson_clocks
#
# Usage:
#   ./build_tensorrt_jetson.sh \
#       --onnx-dir ./artifacts/smolvla_onnx_fp32 \
#       --out-dir ./artifacts/smolvla_trt_orin_nano \
#       --profile orin-nano-8gb
#
#   ./build_tensorrt_jetson.sh --profile orin-nano-8gb --dry-run
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_DIR="${SCRIPT_DIR}/artifacts/smolvla_onnx_fp32"
OUT_DIR=""
PROFILE="orin-nano-8gb"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
DRY_RUN=0
SKIP_BENCH=0
STRICT_CONSTRAINTS=0

usage() {
    cat <<'EOF'
Usage: build_tensorrt_jetson.sh [OPTIONS]

Options:
  --onnx-dir DIR          Directory containing the 3 canonical SmolVLA ONNX files
                          (default: ./artifacts/smolvla_onnx_fp32; FP32 input is
                          recommended so TensorRT applies optimal FP16 tactics
                          with FP32 layerPrecisions directly).
  --out-dir DIR           Output directory for .engine, timing cache, and logs
                          (default: ./artifacts/smolvla_trt_<profile>).
  --profile NAME          Hardware profile:
                            orin-nano-8gb  (JetPack 6.2, 8 GB unified RAM, 1024M workspace)
                            orin-nx-16gb   (JetPack 6.2, 16 GB unified RAM, 2048M workspace)
                            thor           (JetPack 7.2, Blackwell, 4096M workspace)
                          (default: orin-nano-8gb)
  --trtexec PATH          Path to trtexec binary (default: /usr/src/tensorrt/bin/trtexec
                          or PATH lookup if not found there).
  --strict-fp32-norms     Use --precisionConstraints=obey instead of prefer.
  --skip-bench            Build engines only (--iterations=0).
  --dry-run               Print resolved commands without invoking trtexec.
  -h, --help              Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --onnx-dir)
            ONNX_DIR="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --trtexec)
            TRTEXEC="$2"
            shift 2
            ;;
        --strict-fp32-norms)
            STRICT_CONSTRAINTS=1
            shift
            ;;
        --skip-bench)
            SKIP_BENCH=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "${PROFILE}" in
    orin-nano-8gb)
        # Orin Nano 8 GB shares unified LPDDR5 with Ubuntu/JetPack (~1.5 GB) and
        # camera/ROS2 buffers (~0.5 GB). Keep workspace <= 1024M to avoid OOM.
        WORKSPACE_MB=1024
        OPT_LEVEL=3
        TARGET_JETPACK="JetPack 6.2 (TensorRT 10.3+, SM 8.7 Orin Nano 8 GB)"
        EXTRA_PROFILE_FLAGS=()
        ;;
    orin-nx-16gb)
        WORKSPACE_MB=2048
        OPT_LEVEL=4
        TARGET_JETPACK="JetPack 6.2 (TensorRT 10.3+, SM 8.7 Orin NX 16 GB)"
        EXTRA_PROFILE_FLAGS=()
        ;;
    thor)
        WORKSPACE_MB=4096
        OPT_LEVEL=5
        TARGET_JETPACK="JetPack 7.2 (TensorRT 10.x+, Blackwell Jetson Thor)"
        EXTRA_PROFILE_FLAGS=()
        ;;
    *)
        echo "[ERROR] Unsupported --profile '${PROFILE}'. Choose: orin-nano-8gb | orin-nx-16gb | thor" >&2
        exit 2
        ;;
esac

if [[ -z "${OUT_DIR}" ]]; then
    OUT_DIR="${SCRIPT_DIR}/artifacts/smolvla_trt_${PROFILE}"
fi

# Resolve trtexec binary
if [[ "${DRY_RUN}" -eq 0 ]]; then
    if [[ ! -x "${TRTEXEC}" ]]; then
        if command -v trtexec >/dev/null 2>&1; then
            TRTEXEC="$(command -v trtexec)"
        else
            echo "[ERROR] trtexec not found at '${TRTEXEC}' or in PATH." >&2
            echo "        On Jetson JetPack 6.2/7.2, install tensorrt or pass --trtexec." >&2
            exit 1
        fi
    fi
fi

# Preserve FP32 precision on numerically sensitive ops (LayerNorm, RMSNorm,
# Softmax, RoPE Sin/Cos, sinusoidal timestep Sin/Cos, attention mask Where/CumSum).
# TensorRT 10.x trtexec matches wildcard patterns against ONNX layer names.
LAYER_PRECISIONS="*LayerNormalization*:fp32,*layernorm*:fp32,*expert_norm*:fp32,*Softmax*:fp32,*Sin*:fp32,*Cos*:fp32,*Where*:fp32,*CumSum*:fp32"
CONSTRAINT_MODE="prefer"
if [[ "${STRICT_CONSTRAINTS}" -eq 1 ]]; then
    CONSTRAINT_MODE="obey"
fi

TIMING_CACHE="${OUT_DIR}/trt_timing_${PROFILE}.cache"

echo "=============================================================================="
echo "SmolVLA TensorRT 10.x Build"
echo "  Profile:          ${PROFILE} (${TARGET_JETPACK})"
echo "  ONNX Dir:         ${ONNX_DIR}"
echo "  Output Dir:       ${OUT_DIR}"
echo "  Workspace Pool:   ${WORKSPACE_MB}M"
echo "  Opt Level:        ${OPT_LEVEL}"
echo "  Precision Mode:   FP16 + ${CONSTRAINT_MODE} FP32 norms/softmax/RoPE/sinusoid"
echo "  Timing Cache:     ${TIMING_CACHE}"
echo "  Dry Run:          ${DRY_RUN}"
echo "=============================================================================="

if [[ "${DRY_RUN}" -eq 0 ]]; then
    mkdir -p "${OUT_DIR}"
    for req in smolvla_vision_encoder.onnx smolvla_vlm_prefix.onnx smolvla_action_expert_step.onnx; do
        if [[ ! -f "${ONNX_DIR}/${req}" ]]; then
            echo "[ERROR] Missing required ONNX model: ${ONNX_DIR}/${req}" >&2
            exit 1
        fi
    done
fi

build_engine() {
    local name="$1"
    local onnx_file="${ONNX_DIR}/${name}.onnx"
    local engine_file="${OUT_DIR}/${name}.engine"
    local times_file="${OUT_DIR}/${name}_times.json"
    local layers_file="${OUT_DIR}/${name}_layers.json"
    local profile_log="${OUT_DIR}/${name}_build.log"
    local shapes_spec="$2"

    local cmd=(
        "${TRTEXEC}"
        "--onnx=${onnx_file}"
        "--saveEngine=${engine_file}"
        "--fp16"
        "--memPoolSize=workspace:${WORKSPACE_MB}M"
        "--builderOptimizationLevel=${OPT_LEVEL}"
        "--timingCacheFile=${TIMING_CACHE}"
        "--precisionConstraints=${CONSTRAINT_MODE}"
        "--layerPrecisions=${LAYER_PRECISIONS}"
        "--shapes=${shapes_spec}"
        "--dumpProfile"
        "--exportTimes=${times_file}"
        "--exportLayerInfo=${layers_file}"
    )

    if [[ "${SKIP_BENCH}" -eq 1 ]]; then
        cmd+=("--iterations=0" "--warmUp=0" "--duration=0")
    else
        cmd+=("--warmUp=500" "--duration=3" "--iterations=20")
    fi

    if [[ ${#EXTRA_PROFILE_FLAGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_PROFILE_FLAGS[@]}")
    fi

    echo ""
    echo "[build] ${name} -> ${engine_file}"
    echo "        shapes: ${shapes_spec}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '  %q' "${cmd[@]}"
        echo ""
        return 0
    fi

    "${cmd[@]}" 2>&1 | tee "${profile_log}"
    echo "[ok] Built ${engine_file} ($(du -h "${engine_file}" | cut -f1))"
}

# 1. SigLIP Vision Encoder + 4x4 Pixel-Shuffle Connector (3 cameras batched, static 512x512)
build_engine \
    "smolvla_vision_encoder" \
    "pixel_values:3x3x512x512"

# 2. VLM Prefix KV Prefill (16 truncated Llama layers, L_prefix=241)
build_engine \
    "smolvla_vlm_prefix" \
    "image_embeddings:1x192x960,img_masks:1x3,lang_tokens:1x48,lang_masks:1x48,state:1x32"

# 3. Action Expert Single Euler Denoising Step (executed 10x per chunk, L_chunk=50)
build_engine \
    "smolvla_action_expert_step" \
    "x_t:1x50x32,timestep:1,past_keys:16x1x5x241x64,past_values:16x1x5x241x64,prefix_pad_masks:1x241"

echo ""
echo "=============================================================================="
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] All 3 TensorRT build commands validated successfully."
else
    echo "[done] All 3 TensorRT engines built in: ${OUT_DIR}"
    ls -lh "${OUT_DIR}"/*.engine
fi
echo "=============================================================================="
