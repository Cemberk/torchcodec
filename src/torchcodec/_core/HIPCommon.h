// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// HIP equivalents of the utilities in CUDACommon.h.
// On ROCm, PyTorch reuses the torch::kCUDA device type for HIP devices,
// and cudaStream_t maps to hipStream_t. This file provides the ROCm-specific
// implementations for device initialization, color conversion, and validation.

#pragma once

// On ROCm, we avoid including ATen/cuda headers directly with GCC because
// they reference <cuda.h> which is only available when compiling with hipcc.
// Instead, we use HIP types directly and convert at the boundary.
#include <torch/types.h>

#include "FFMPEGCommon.h"
#include "Frame.h"

// Use the HIP-compatible types from the ROCm installation.
// hipStream_t and other HIP types are defined here.
#include <hip/hip_runtime.h>

namespace facebook::torchcodec {

// Maximum number of GPUs supported (same as PyTorch's limit)
constexpr int MAX_ROCM_GPUS = 128;

// Initialize the HIP/ROCm context through PyTorch to ensure compatibility.
// This must be called before any ROCm operations.
void initializeHIPContextWithPytorch(const torch::Device& device);

// Convert an NV12 frame (in GPU memory) to an RGB HWC tensor on the same GPU.
// This uses custom HIP kernels instead of NPP.
torch::Tensor convertNV12FrameToRGB_HIP(
    uint8_t* nv12Data,
    int nv12Pitch,
    int width,
    int height,
    int colorspace, // AVColorSpace enum value
    int colorRange, // AVColorRange enum value
    const torch::Device& device,
    hipStream_t stream,
    std::optional<torch::Tensor> preAllocatedOutputTensor = std::nullopt);

// Convert an RGB HWC tensor to NV12 format in GPU memory (for encoding).
// Returns a pair of device pointers: {Y plane, UV plane} and the pitch.
struct NV12Frame {
  uint8_t* yPlane;
  uint8_t* uvPlane;
  int pitch;
  size_t totalSize;
};

NV12Frame convertRGBToNV12_HIP(
    const torch::Tensor& rgbTensor,
    int colorspace,
    int colorRange,
    hipStream_t stream);

// Validate pre-allocated tensor shape matches expected frame dimensions
void validatePreAllocatedTensorShape_HIP(
    const std::optional<torch::Tensor>& preAllocatedOutputTensor,
    int width,
    int height);

// Get device index, handling the -1 (current device) case
int getDeviceIndex_HIP(const torch::Device& device);

// HIP kernel launch functions (implemented in HIPColorspaceKernels.hip)
void launchNv12ToRgb24Kernel(
    const uint8_t* nv12,
    int nv12Pitch,
    uint8_t* rgb,
    int rgbPitch,
    int width,
    int height,
    int colorStandard,
    hipStream_t stream);

void launchRgb24ToNv12Kernel(
    const uint8_t* rgb,
    int rgbPitch,
    uint8_t* nv12Y,
    int yPitch,
    uint8_t* nv12UV,
    int uvPitch,
    int width,
    int height,
    int colorStandard,
    hipStream_t stream);

} // namespace facebook::torchcodec
