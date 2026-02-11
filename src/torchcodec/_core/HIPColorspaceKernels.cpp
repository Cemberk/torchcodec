// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// HIP kernels for YUV<->RGB color conversion on AMD GPUs.
// These replace the NPP library functions used in the CUDA path.
// The color conversion math matches the CPU path (FFmpeg) as closely
// as possible for consistency.
//
// Adapted from rocDecode's colorspace_kernels.cpp with modifications
// for TorchCodec's output format requirements (HWC uint8 RGB24).

// This file contains HIP GPU kernels. It is compiled as C++ but uses
// HIP intrinsics (__global__, threadIdx, etc.) which are available
// when compiling with the ROCm Clang compiler and -x hip flag.
// See CMakeLists.txt for the compile flags setup.
//
// NOTE: This file must NOT include HIPCommon.h or any ATen/torch headers
// because those pull in <cuda.h> which is not available when compiling
// with the standalone ROCm Clang. Only <hip/hip_runtime.h> is needed.
#include <hip/hip_runtime.h>
#include <cstdint>

namespace facebook::torchcodec {

// ---- Constant memory for color conversion matrices ----
__constant__ float d_yuv2rgb_mat[3][3];
__constant__ float d_rgb2yuv_mat[3][3];

// ---- Helper: compute color matrix coefficients ----
// Color space standard values match AVColorSpace enum values used in FFmpeg
enum HIPColorStandard {
  HIP_CS_BT709 = 1,
  HIP_CS_FCC = 4,
  HIP_CS_BT470 = 5,
  HIP_CS_BT601 = 6,
  HIP_CS_SMPTE240M = 7,
  HIP_CS_BT2020 = 9,
};

static void getColorMatCoefficients(
    int colStandard,
    float& wr,
    float& wb,
    int& black,
    int& white,
    int& maxVal) {
  black = 16;
  white = 235;
  maxVal = 255;

  switch (colStandard) {
    case HIP_CS_BT709:
    default:
      wr = 0.2126f;
      wb = 0.0722f;
      break;
    case HIP_CS_FCC:
      wr = 0.30f;
      wb = 0.11f;
      break;
    case HIP_CS_BT470:
    case HIP_CS_BT601:
      wr = 0.2990f;
      wb = 0.1140f;
      break;
    case HIP_CS_SMPTE240M:
      wr = 0.212f;
      wb = 0.087f;
      break;
    case HIP_CS_BT2020:
      wr = 0.2627f;
      wb = 0.0593f;
      break;
  }
}

static void setMatYuv2Rgb(int colStandard, hipStream_t stream) {
  float wr, wb;
  int black, white, maxVal;
  getColorMatCoefficients(colStandard, wr, wb, black, white, maxVal);
  float mat[3][3] = {
      {1.0f, 0.0f, (1.0f - wr) / 0.5f},
      {1.0f,
       -wb * (1.0f - wb) / 0.5f / (1.0f - wb - wr),
       -wr * (1.0f - wr) / 0.5f / (1.0f - wb - wr)},
      {1.0f, (1.0f - wb) / 0.5f, 0.0f},
  };
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      mat[i][j] = static_cast<float>(
          1.0 * maxVal / (white - black) * mat[i][j]);
    }
  }
  hipMemcpyToSymbolAsync(
      d_yuv2rgb_mat, mat, sizeof(mat), 0, hipMemcpyHostToDevice, stream);
}

static void setMatRgb2Yuv(int colStandard, hipStream_t stream) {
  float wr, wb;
  int black, white, maxVal;
  getColorMatCoefficients(colStandard, wr, wb, black, white, maxVal);
  float mat[3][3] = {
      {wr, 1.0f - wb - wr, wb},
      {-0.5f * wr / (1.0f - wb),
       -0.5f * (1.0f - wb - wr) / (1.0f - wb),
       0.5f},
      {0.5f,
       -0.5f * (1.0f - wb - wr) / (1.0f - wr),
       -0.5f * wb / (1.0f - wr)},
  };
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      mat[i][j] = static_cast<float>(
          1.0 * (white - black) / maxVal * mat[i][j]);
    }
  }
  hipMemcpyToSymbolAsync(
      d_rgb2yuv_mat, mat, sizeof(mat), 0, hipMemcpyHostToDevice, stream);
}

// ---- Device helpers ----
template <class T>
__device__ static T clampVal(T x, T lower, T upper) {
  return x < lower ? lower : (x > upper ? upper : x);
}

// ---- NV12 -> RGB24 kernel ----
// NV12 layout: Y plane (height rows of width bytes at pitch stride),
//              UV plane (height/2 rows of width bytes, interleaved U,V)
// Output: HWC uint8 RGB (3 bytes per pixel, contiguous or strided)
__global__ void nv12ToRgb24Kernel(
    const uint8_t* __restrict__ nv12,
    int nv12Pitch,
    uint8_t* __restrict__ rgb,
    int rgbPitch,
    int width,
    int height,
    int vPitch) {
  // Each thread processes one pixel
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= width || y >= height) {
    return;
  }

  // Y value
  float fy = static_cast<float>(nv12[y * nv12Pitch + x]) - 16.0f;
  // UV values (subsampled 2x2)
  int uvOffset = vPitch * nv12Pitch + (y / 2) * nv12Pitch + (x & ~1);
  float fu = static_cast<float>(nv12[uvOffset]) - 128.0f;
  float fv = static_cast<float>(nv12[uvOffset + 1]) - 128.0f;

  float r = d_yuv2rgb_mat[0][0] * fy + d_yuv2rgb_mat[0][1] * fu +
      d_yuv2rgb_mat[0][2] * fv;
  float g = d_yuv2rgb_mat[1][0] * fy + d_yuv2rgb_mat[1][1] * fu +
      d_yuv2rgb_mat[1][2] * fv;
  float b = d_yuv2rgb_mat[2][0] * fy + d_yuv2rgb_mat[2][1] * fu +
      d_yuv2rgb_mat[2][2] * fv;

  int outIdx = y * rgbPitch + x * 3;
  rgb[outIdx + 0] = static_cast<uint8_t>(clampVal(r, 0.0f, 255.0f));
  rgb[outIdx + 1] = static_cast<uint8_t>(clampVal(g, 0.0f, 255.0f));
  rgb[outIdx + 2] = static_cast<uint8_t>(clampVal(b, 0.0f, 255.0f));
}

// ---- RGB24 -> NV12 kernel (for encoding support) ----
// Input: HWC uint8 RGB (3 bytes per pixel)
// Output: NV12 (Y plane + interleaved UV plane)
__global__ void rgb24ToNv12Kernel(
    const uint8_t* __restrict__ rgb,
    int rgbPitch,
    uint8_t* __restrict__ nv12Y,
    int yPitch,
    uint8_t* __restrict__ nv12UV,
    int uvPitch,
    int width,
    int height) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= width || y >= height) {
    return;
  }

  int rgbIdx = y * rgbPitch + x * 3;
  float r = static_cast<float>(rgb[rgbIdx + 0]);
  float g = static_cast<float>(rgb[rgbIdx + 1]);
  float b = static_cast<float>(rgb[rgbIdx + 2]);

  // Y
  float yVal = d_rgb2yuv_mat[0][0] * r + d_rgb2yuv_mat[0][1] * g +
      d_rgb2yuv_mat[0][2] * b + 16.0f;
  nv12Y[y * yPitch + x] = static_cast<uint8_t>(clampVal(yVal, 0.0f, 255.0f));

  // UV (only for even coordinates - 2x2 subsampling)
  if ((x & 1) == 0 && (y & 1) == 0) {
    // Average 2x2 block for chroma
    float rAvg = r, gAvg = g, bAvg = b;
    int count = 1;
    if (x + 1 < width) {
      int idx2 = y * rgbPitch + (x + 1) * 3;
      rAvg += static_cast<float>(rgb[idx2 + 0]);
      gAvg += static_cast<float>(rgb[idx2 + 1]);
      bAvg += static_cast<float>(rgb[idx2 + 2]);
      count++;
    }
    if (y + 1 < height) {
      int idx3 = (y + 1) * rgbPitch + x * 3;
      rAvg += static_cast<float>(rgb[idx3 + 0]);
      gAvg += static_cast<float>(rgb[idx3 + 1]);
      bAvg += static_cast<float>(rgb[idx3 + 2]);
      count++;
    }
    if (x + 1 < width && y + 1 < height) {
      int idx4 = (y + 1) * rgbPitch + (x + 1) * 3;
      rAvg += static_cast<float>(rgb[idx4 + 0]);
      gAvg += static_cast<float>(rgb[idx4 + 1]);
      bAvg += static_cast<float>(rgb[idx4 + 2]);
      count++;
    }
    rAvg /= count;
    gAvg /= count;
    bAvg /= count;

    float uVal = d_rgb2yuv_mat[1][0] * rAvg + d_rgb2yuv_mat[1][1] * gAvg +
        d_rgb2yuv_mat[1][2] * bAvg + 128.0f;
    float vVal = d_rgb2yuv_mat[2][0] * rAvg + d_rgb2yuv_mat[2][1] * gAvg +
        d_rgb2yuv_mat[2][2] * bAvg + 128.0f;

    int uvIdx = (y / 2) * uvPitch + x;
    nv12UV[uvIdx] = static_cast<uint8_t>(clampVal(uVal, 0.0f, 255.0f));
    nv12UV[uvIdx + 1] = static_cast<uint8_t>(clampVal(vVal, 0.0f, 255.0f));
  }
}

// ---- Host launch functions ----

void launchNv12ToRgb24Kernel(
    const uint8_t* nv12,
    int nv12Pitch,
    uint8_t* rgb,
    int rgbPitch,
    int width,
    int height,
    int colorStandard,
    hipStream_t stream) {
  setMatYuv2Rgb(colorStandard, stream);

  dim3 block(16, 16);
  dim3 grid(
      (width + block.x - 1) / block.x,
      (height + block.y - 1) / block.y);

  // vPitch = height (the UV plane starts after height rows of Y data)
  nv12ToRgb24Kernel<<<grid, block, 0, stream>>>(
      nv12, nv12Pitch, rgb, rgbPitch, width, height, height);
}

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
    hipStream_t stream) {
  setMatRgb2Yuv(colorStandard, stream);

  dim3 block(16, 16);
  dim3 grid(
      (width + block.x - 1) / block.x,
      (height + block.y - 1) / block.y);

  rgb24ToNv12Kernel<<<grid, block, 0, stream>>>(
      rgb, rgbPitch, nv12Y, yPitch, nv12UV, uvPitch, width, height);
}

} // namespace facebook::torchcodec
