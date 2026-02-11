// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "RocDecodeCache.h"

namespace facebook::torchcodec {

RocDecodeCache& RocDecodeCache::getCache(const torch::Device& device) {
  static RocDecodeCache cacheInstances[MAX_ROCM_GPUS];
  return cacheInstances[getDeviceIndex_HIP(device)];
}

UniqueRocDecoder RocDecodeCache::getDecoder(
    const RocdecVideoFormat* videoFormat) {
  CacheKey key(videoFormat);
  std::lock_guard<std::mutex> lock(cacheLock_);

  auto it = cache_.find(key);
  if (it != cache_.end()) {
    auto decoder = std::move(it->second);
    cache_.erase(it);
    return decoder;
  }

  return nullptr;
}

bool RocDecodeCache::returnDecoder(
    const RocdecVideoFormat* videoFormat,
    UniqueRocDecoder decoder) {
  if (!decoder) {
    return false;
  }

  CacheKey key(videoFormat);
  std::lock_guard<std::mutex> lock(cacheLock_);

  if (cache_.size() >= MAX_CACHE_SIZE) {
    return false;
  }

  cache_[key] = std::move(decoder);
  return true;
}

} // namespace facebook::torchcodec
