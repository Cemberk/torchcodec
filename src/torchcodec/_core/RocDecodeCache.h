// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Cache for rocDecode decoder instances, analogous to NVDECCache.h.
// Decoder creation is expensive, so we cache and reuse decoders with
// matching parameters across VideoDecoder instances.

#pragma once

#include <map>
#include <memory>
#include <mutex>

#include <torch/types.h>

#include "HIPCommon.h"
#include "RocDecodeRuntimeLoader.h"
#include "rocdecode_include/rocdecode.h"
#include "rocdecode_include/rocparser.h"

namespace facebook::torchcodec {

// Custom deleter for rocDecode decoder handles
struct RocDecoderDeleter {
  void operator()(rocDecDecoderHandle* handlePtr) const {
    if (handlePtr && *handlePtr) {
      rocDecDestroyDecoder(*handlePtr);
      delete handlePtr;
    }
  }
};

using UniqueRocDecoder =
    std::unique_ptr<rocDecDecoderHandle, RocDecoderDeleter>;

// A per-device cache for rocDecode decoders. There is one instance of this
// class per GPU device, and it is accessed through the static getCache() method.
class RocDecodeCache {
 public:
  static RocDecodeCache& getCache(const torch::Device& device);

  // Get decoder from cache - returns nullptr if none available
  UniqueRocDecoder getDecoder(const RocdecVideoFormat* videoFormat);

  // Return decoder to cache - returns true if added to cache
  bool returnDecoder(
      const RocdecVideoFormat* videoFormat,
      UniqueRocDecoder decoder);

 private:
  // Cache key struct: a decoder can be reused only if all these parameters match
  struct CacheKey {
    rocDecVideoCodec codecType;
    uint32_t width;
    uint32_t height;
    rocDecVideoChromaFormat chromaFormat;
    uint32_t bitDepthLumaMinus8;
    uint8_t numDecodeSurfaces;

    CacheKey() = delete;

    explicit CacheKey(const RocdecVideoFormat* videoFormat)
        : codecType(videoFormat->codec),
          width(videoFormat->coded_width),
          height(videoFormat->coded_height),
          chromaFormat(videoFormat->chroma_format),
          bitDepthLumaMinus8(videoFormat->bit_depth_luma_minus8),
          numDecodeSurfaces(videoFormat->min_num_decode_surfaces) {}

    CacheKey(const CacheKey&) = default;
    CacheKey& operator=(const CacheKey&) = default;

    bool operator<(const CacheKey& other) const {
      return std::tie(
                 codecType,
                 width,
                 height,
                 chromaFormat,
                 bitDepthLumaMinus8,
                 numDecodeSurfaces) <
          std::tie(
                 other.codecType,
                 other.width,
                 other.height,
                 other.chromaFormat,
                 other.bitDepthLumaMinus8,
                 other.numDecodeSurfaces);
    }
  };

  RocDecodeCache() = default;
  ~RocDecodeCache() = default;

  std::map<CacheKey, UniqueRocDecoder> cache_;
  std::mutex cacheLock_;

  // Max number of cached decoders per device
  static constexpr int MAX_CACHE_SIZE = 20;
};

} // namespace facebook::torchcodec
