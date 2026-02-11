// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "HIPCommon.h"

extern "C" {
#include <libavutil/pixfmt.h>
}

namespace facebook::torchcodec {

void initializeHIPContextWithPytorch(const torch::Device& device) {
  // It is important for PyTorch itself to create the HIP/ROCm context.
  // If some other library creates the context it may not be compatible
  // with PyTorch. This is a dummy tensor to initialize the context.
  // On ROCm PyTorch, torch::kCUDA maps to HIP devices.
  torch::Tensor dummyTensorForHIPInitialization = torch::zeros(
      {1}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
}

// Map AVColorSpace to the color standard enum used by our HIP kernels.
// These values match the matrix_coefficients field in video signal description.
static int avColorSpaceToHIPStandard(int colorspace) {
  switch (colorspace) {
    case AVCOL_SPC_BT709:
      return 1; // HIP_CS_BT709
    case AVCOL_SPC_BT470BG:
    case AVCOL_SPC_SMPTE170M:
      return 6; // HIP_CS_BT601
    case AVCOL_SPC_BT2020_NCL:
    case AVCOL_SPC_BT2020_CL:
      return 9; // HIP_CS_BT2020
    case AVCOL_SPC_SMPTE240M:
      return 7; // HIP_CS_SMPTE240M
    case AVCOL_SPC_FCC:
      return 4; // HIP_CS_FCC
    default:
      return 6; // Default to BT.601
  }
}

torch::Tensor convertNV12FrameToRGB_HIP(
    uint8_t* nv12Data,
    int nv12Pitch,
    int width,
    int height,
    int colorspace,
    int colorRange,
    const torch::Device& device,
    hipStream_t stream,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  auto frameDims = FrameDims(height, width);
  torch::Tensor dst;
  if (preAllocatedOutputTensor.has_value()) {
    dst = preAllocatedOutputTensor.value();
  } else {
    dst = allocateEmptyHWCTensor(frameDims, device);
  }

  int colStandard = avColorSpaceToHIPStandard(colorspace);

  // The RGB output pitch is the tensor's row stride in bytes
  // dst shape is [H, W, 3], stride(0) gives the row stride
  int rgbPitch = static_cast<int>(dst.stride(0) * dst.element_size());

  launchNv12ToRgb24Kernel(
      nv12Data,
      nv12Pitch,
      static_cast<uint8_t*>(dst.data_ptr()),
      rgbPitch,
      width,
      height,
      colStandard,
      stream);

  return dst;
}

NV12Frame convertRGBToNV12_HIP(
    const torch::Tensor& rgbTensor,
    int colorspace,
    [[maybe_unused]] int colorRange,
    hipStream_t stream) {
  TORCH_CHECK(
      rgbTensor.dim() == 3 && rgbTensor.size(2) == 3,
      "Expected HWC RGB tensor with 3 channels, got shape: ",
      rgbTensor.sizes());
  TORCH_CHECK(
      rgbTensor.device().type() == torch::kCUDA,
      "Expected tensor on CUDA/HIP device, got: ",
      rgbTensor.device().str());

  int height = static_cast<int>(rgbTensor.size(0));
  int width = static_cast<int>(rgbTensor.size(1));
  int rgbPitch = static_cast<int>(rgbTensor.stride(0) * rgbTensor.element_size());

  // Allocate NV12 buffer: Y plane (width * height) + UV plane (width * height/2)
  int ySize = width * height;
  int uvSize = width * (height / 2);
  size_t totalSize = static_cast<size_t>(ySize + uvSize);

  uint8_t* nv12Buffer = nullptr;
  hipError_t err = hipMalloc(
      reinterpret_cast<void**>(&nv12Buffer), totalSize);
  TORCH_CHECK(
      err == hipSuccess,
      "Failed to allocate HIP memory for NV12: ",
      hipGetErrorString(err));

  uint8_t* yPlane = nv12Buffer;
  uint8_t* uvPlane = nv12Buffer + ySize;

  int colStandard = avColorSpaceToHIPStandard(colorspace);

  launchRgb24ToNv12Kernel(
      static_cast<const uint8_t*>(rgbTensor.data_ptr()),
      rgbPitch,
      yPlane,
      width, // Y pitch = width (tightly packed)
      uvPlane,
      width, // UV pitch = width
      width,
      height,
      colStandard,
      stream);

  return NV12Frame{yPlane, uvPlane, width, totalSize};
}

void validatePreAllocatedTensorShape_HIP(
    const std::optional<torch::Tensor>& preAllocatedOutputTensor,
    int width,
    int height) {
  if (preAllocatedOutputTensor.has_value()) {
    auto shape = preAllocatedOutputTensor.value().sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == height) &&
            (shape[1] == width) && (shape[2] == 3),
        "Expected tensor of shape ",
        height,
        "x",
        width,
        "x3, got ",
        shape);
  }
}

int getDeviceIndex_HIP(const torch::Device& device) {
  int deviceIndex = static_cast<int>(device.index());
  TORCH_CHECK(
      deviceIndex >= -1 && deviceIndex < MAX_ROCM_GPUS,
      "Invalid device index = ",
      deviceIndex);

  if (deviceIndex == -1) {
    // On ROCm PyTorch, hipGetDevice works through the CUDA->HIP mapping
    TORCH_CHECK(
        hipGetDevice(&deviceIndex) == hipSuccess,
        "Failed to get current HIP device.");
  }
  return deviceIndex;
}

// Provide the getDeviceIndex() definition required by Cache.h's
// PerGpuCache template. On CUDA builds this is defined in CUDACommon.cpp;
// on ROCm builds we provide the equivalent here.
int getDeviceIndex(const torch::Device& device) {
  return getDeviceIndex_HIP(device);
}

} // namespace facebook::torchcodec
