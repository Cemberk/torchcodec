#!/bin/bash
# =============================================================================
# TorchCodec GPU Benchmark Runner
# =============================================================================
#
# Builds a Docker image (ROCm or CUDA) and runs the GPU benchmark inside it.
# Produces JSON results for cross-platform comparison.
#
# Usage:
#   # Auto-detect GPU vendor and run
#   bash benchmarks/run_gpu_benchmark.sh
#
#   # Force ROCm build for MI300X
#   bash benchmarks/run_gpu_benchmark.sh --rocm --arch gfx942
#
#   # Force CUDA build
#   bash benchmarks/run_gpu_benchmark.sh --cuda
#
#   # Quick benchmark (fewer iterations)
#   bash benchmarks/run_gpu_benchmark.sh --quick
#
#   # Test multiple codecs
#   bash benchmarks/run_gpu_benchmark.sh --codecs h264,h265,av1
#
#   # Skip Docker build (run directly if torchcodec is already installed)
#   bash benchmarks/run_gpu_benchmark.sh --no-docker
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
GPU_VENDOR=""
HIP_ARCH=""
PYTORCH_ROCM_IMAGE="rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.8.0"
CUDA_VERSION="12.6.3"
CODECS="h264"
QUICK=""
THREADS=1
SYNTHETIC=""
NO_DOCKER=false
OUTPUT_DIR="${REPO_ROOT}/benchmark_results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --rocm)       GPU_VENDOR="AMD";    shift ;;
        --cuda)       GPU_VENDOR="NVIDIA"; shift ;;
        --arch)       HIP_ARCH="$2";       shift 2 ;;
        --rocm-image) PYTORCH_ROCM_IMAGE="$2"; shift 2 ;;
        --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
        --codecs)     CODECS="$2";         shift 2 ;;
        --quick)      QUICK="--quick";     shift ;;
        --threads)    THREADS="$2";        shift 2 ;;
        --synthetic)  SYNTHETIC="--synthetic"; shift ;;
        --no-docker)  NO_DOCKER=true;      shift ;;
        --output-dir) OUTPUT_DIR="$2";     shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --rocm              Force ROCm (AMD) build"
            echo "  --cuda              Force CUDA (NVIDIA) build"
            echo "  --arch ARCH         HIP architecture (e.g., gfx942 for MI300X)"
            echo "  --rocm-image IMAGE  rocm/pytorch base image tag"
            echo "                      (default: rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.8.0)"
            echo "  --cuda-version VER  CUDA version (default: 12.6.3)"
            echo "  --codecs LIST       Comma-separated codecs (default: h264)"
            echo "  --quick             Quick mode (fewer iterations)"
            echo "  --threads N         Multi-threaded decode threads (default: 1)"
            echo "  --synthetic         Generate and test synthetic videos"
            echo "  --no-docker         Run directly without Docker"
            echo "  --output-dir DIR    Output directory (default: benchmark_results/)"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Auto-detect GPU vendor if not specified
if [[ -z "$GPU_VENDOR" ]]; then
    if command -v rocminfo &>/dev/null && rocminfo 2>/dev/null | grep -q "gfx"; then
        GPU_VENDOR="AMD"
        echo "[*] Auto-detected AMD GPU (ROCm)"
    elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        GPU_VENDOR="NVIDIA"
        echo "[*] Auto-detected NVIDIA GPU (CUDA)"
    else
        echo "ERROR: Could not detect GPU. Use --rocm or --cuda to specify."
        exit 1
    fi
fi

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Docker mode
# ---------------------------------------------------------------------------
if [[ "$NO_DOCKER" == false ]]; then
    cd "$REPO_ROOT"

    if [[ "$GPU_VENDOR" == "AMD" ]]; then
        # Derive a short tag from the image name for the local build
        IMAGE_TAG="torchcodec-bench:$(echo "$PYTORCH_ROCM_IMAGE" | sed 's|rocm/pytorch:||; s|_|-|g')"
        echo "[*] Building ROCm image: $IMAGE_TAG"
        echo "[*] Base image: $PYTORCH_ROCM_IMAGE"

        BUILD_ARGS=(
            --build-arg "PYTORCH_ROCM_IMAGE=${PYTORCH_ROCM_IMAGE}"
        )
        if [[ -n "$HIP_ARCH" ]]; then
            BUILD_ARGS+=(--build-arg "HIP_ARCHITECTURES=${HIP_ARCH}")
        fi

        docker build -f docker/Dockerfile.rocm \
            "${BUILD_ARGS[@]}" \
            -t "$IMAGE_TAG" .

        echo "[*] Running benchmark in ROCm container..."
        docker run --rm \
            --device /dev/kfd --device /dev/dri \
            --group-add video \
            -v "$OUTPUT_DIR:/results" \
            "$IMAGE_TAG" \
            python3 /workspace/benchmarks/decoders/rocm_vs_cuda_benchmark.py \
                --codecs "$CODECS" \
                --threads "$THREADS" \
                --output "/results/benchmark_rocm.json" \
                $QUICK $SYNTHETIC

    elif [[ "$GPU_VENDOR" == "NVIDIA" ]]; then
        IMAGE_TAG="torchcodec-bench:cuda${CUDA_VERSION}"
        echo "[*] Building CUDA image: $IMAGE_TAG"

        docker build -f docker/Dockerfile.cuda \
            --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
            -t "$IMAGE_TAG" .

        echo "[*] Running benchmark in CUDA container..."
        docker run --rm \
            --gpus all \
            -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
            -v "$OUTPUT_DIR:/results" \
            "$IMAGE_TAG" \
            python3 /workspace/benchmarks/decoders/rocm_vs_cuda_benchmark.py \
                --codecs "$CODECS" \
                --threads "$THREADS" \
                --output "/results/benchmark_cuda.json" \
                $QUICK $SYNTHETIC
    fi

# ---------------------------------------------------------------------------
# Non-Docker mode (run directly)
# ---------------------------------------------------------------------------
else
    echo "[*] Running benchmark directly (no Docker)..."
    cd "$REPO_ROOT"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    VENDOR_LOWER=$(echo "$GPU_VENDOR" | tr '[:upper:]' '[:lower:]')
    OUTPUT_FILE="${OUTPUT_DIR}/benchmark_${VENDOR_LOWER}_${TIMESTAMP}.json"

    python3 benchmarks/decoders/rocm_vs_cuda_benchmark.py \
        --codecs "$CODECS" \
        --threads "$THREADS" \
        --output "$OUTPUT_FILE" \
        $QUICK $SYNTHETIC
fi

echo ""
echo "[*] Benchmark complete. Results in: $OUTPUT_DIR/"
echo ""

# If both ROCm and CUDA results exist, print comparison
ROCM_FILE="$OUTPUT_DIR/benchmark_rocm.json"
CUDA_FILE="$OUTPUT_DIR/benchmark_cuda.json"

if [[ -f "$ROCM_FILE" && -f "$CUDA_FILE" ]]; then
    echo "============================================================"
    echo "  Both ROCm and CUDA results found! Quick comparison:"
    echo "============================================================"
    python3 -c "
import json, sys

with open('$ROCM_FILE') as f:
    rocm = json.load(f)
with open('$CUDA_FILE') as f:
    cuda = json.load(f)

print(f\"  ROCm GPU: {rocm['gpu_info']['name']}\")
print(f\"  CUDA GPU: {cuda['gpu_info']['name']}\")
print()
print(f\"{'Test':<30} {'ROCm FPS':>10} {'CUDA FPS':>10} {'Ratio':>8}\")
print('-' * 60)

rocm_by_key = {}
for r in rocm['results']:
    if r['device'] == 'gpu':
        rocm_by_key[(r['test_name'], r['codec'])] = r['decode_fps_median']

cuda_by_key = {}
for r in cuda['results']:
    if r['device'] == 'gpu':
        cuda_by_key[(r['test_name'], r['codec'])] = r['decode_fps_median']

for key in sorted(set(rocm_by_key) | set(cuda_by_key)):
    r_fps = rocm_by_key.get(key, 0)
    c_fps = cuda_by_key.get(key, 0)
    ratio = r_fps / c_fps if c_fps > 0 else float('inf')
    label = f\"{key[0]} ({key[1]})\"
    print(f\"{label:<30} {r_fps:>10.1f} {c_fps:>10.1f} {ratio:>7.2f}x\")
print()
"
fi
