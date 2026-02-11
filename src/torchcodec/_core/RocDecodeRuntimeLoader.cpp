// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "RocDecodeRuntimeLoader.h"

#include "rocdecode_include/rocdecode.h"
#include "rocdecode_include/rocparser.h"

#include <torch/types.h>
#include <cstdio>
#include <mutex>

#include <dlfcn.h>
typedef void* tHandle;

namespace facebook::torchcodec {

/* clang-format off */
// This file defines the logic to load the rocDecode library **at runtime**,
// along with the corresponding rocDecode functions that we'll need.
//
// We do this because we *do not want* to link (statically or dynamically)
// against librocdecode.so: it is not always available on the users machine!
// If we were to link against librocdecode.so, that would mean that our
// libtorchcodec_coreN.so would try to look for it when loaded at import time.
// And if it's not on the users machine, that causes `import torchcodec` to
// fail.
//
// This mirrors exactly the pattern used in NVCUVIDRuntimeLoader.cpp for
// libnvcuvid.so. See that file for a detailed explanation of the technique.
//
// At runtime, when the RocDecode device interface is first created, we call
// loadRocDecodeLibrary() which dlopen()s librocdecode.so and binds all
// function pointers via dlsym(). If the library is not found, we fall back
// to CPU decoding.

// ---- Function pointer types for rocDecode API ----
typedef rocDecStatus ROCDECAPI trocDecCreateDecoder(rocDecDecoderHandle*, RocDecoderCreateInfo*);
typedef rocDecStatus ROCDECAPI trocDecDestroyDecoder(rocDecDecoderHandle);
typedef rocDecStatus ROCDECAPI trocDecGetDecoderCaps(RocdecDecodeCaps*);
typedef rocDecStatus ROCDECAPI trocDecDecodeFrame(rocDecDecoderHandle, RocdecPicParams*);
typedef rocDecStatus ROCDECAPI trocDecGetDecodeStatus(rocDecDecoderHandle, int, RocdecDecodeStatus*);
typedef rocDecStatus ROCDECAPI trocDecReconfigureDecoder(rocDecDecoderHandle, RocdecReconfigureDecoderInfo*);
typedef rocDecStatus ROCDECAPI trocDecGetVideoFrame(rocDecDecoderHandle, int, void*[3], uint32_t*, RocdecProcParams*);
typedef const char*  ROCDECAPI trocDecGetErrorName(rocDecStatus);

// ---- Function pointer types for rocParser API ----
typedef rocDecStatus ROCDECAPI trocDecCreateVideoParser(RocdecVideoParser*, RocdecParserParams*);
typedef rocDecStatus ROCDECAPI trocDecParseVideoData(RocdecVideoParser, RocdecSourceDataPacket*);
typedef rocDecStatus ROCDECAPI trocDecDestroyVideoParser(RocdecVideoParser);
/* clang-format on */

// Global function pointers - will be dynamically loaded
static trocDecCreateDecoder* dl_rocDecCreateDecoder = nullptr;
static trocDecDestroyDecoder* dl_rocDecDestroyDecoder = nullptr;
static trocDecGetDecoderCaps* dl_rocDecGetDecoderCaps = nullptr;
static trocDecDecodeFrame* dl_rocDecDecodeFrame = nullptr;
static trocDecGetDecodeStatus* dl_rocDecGetDecodeStatus = nullptr;
static trocDecReconfigureDecoder* dl_rocDecReconfigureDecoder = nullptr;
static trocDecGetVideoFrame* dl_rocDecGetVideoFrame = nullptr;
static trocDecGetErrorName* dl_rocDecGetErrorName = nullptr;

static trocDecCreateVideoParser* dl_rocDecCreateVideoParser = nullptr;
static trocDecParseVideoData* dl_rocDecParseVideoData = nullptr;
static trocDecDestroyVideoParser* dl_rocDecDestroyVideoParser = nullptr;

static tHandle g_rocdecode_handle = nullptr;
static tHandle g_rocparser_handle = nullptr;
static std::mutex g_rocdecode_mutex;

bool isLoaded() {
  return (
      g_rocdecode_handle && dl_rocDecCreateDecoder && dl_rocDecDestroyDecoder &&
      dl_rocDecGetDecoderCaps && dl_rocDecDecodeFrame &&
      dl_rocDecGetVideoFrame && dl_rocDecGetErrorName &&
      dl_rocDecCreateVideoParser && dl_rocDecParseVideoData &&
      dl_rocDecDestroyVideoParser);
}

template <typename T>
T* bindFunction(tHandle handle, const char* functionName) {
  return reinterpret_cast<T*>(dlsym(handle, functionName));
}

bool _loadLibraries() {
  // Try versioned names first (future-proof for soname bumps),
  // then unversioned. This ensures we pick up whatever rocDecode
  // is installed regardless of the ROCm version.
  const char* decoder_lib_names[] = {
      "librocdecode.so",   // unversioned (typical for dev installs)
      "librocdecode.so.0", // soversion 0
      "librocdecode.so.1", // future soversion bump
      nullptr};

  for (const char** name = decoder_lib_names; *name != nullptr; ++name) {
    g_rocdecode_handle = dlopen(*name, RTLD_NOW);
    if (g_rocdecode_handle != nullptr) {
      break;
    }
  }
  if (g_rocdecode_handle == nullptr) {
    return false;
  }

  // The parser functions may be in the same library or a separate one.
  // Try the main library first, then try a separate parser library.
  g_rocparser_handle = g_rocdecode_handle;

  return true;
}

bool loadRocDecodeLibrary() {
  std::lock_guard<std::mutex> lock(g_rocdecode_mutex);

  if (isLoaded()) {
    return true;
  }

  if (!_loadLibraries()) {
    return false;
  }

  // Load decoder function pointers
  dl_rocDecCreateDecoder =
      bindFunction<trocDecCreateDecoder>(g_rocdecode_handle, "rocDecCreateDecoder");
  dl_rocDecDestroyDecoder =
      bindFunction<trocDecDestroyDecoder>(g_rocdecode_handle, "rocDecDestroyDecoder");
  dl_rocDecGetDecoderCaps =
      bindFunction<trocDecGetDecoderCaps>(g_rocdecode_handle, "rocDecGetDecoderCaps");
  dl_rocDecDecodeFrame =
      bindFunction<trocDecDecodeFrame>(g_rocdecode_handle, "rocDecDecodeFrame");
  dl_rocDecGetDecodeStatus =
      bindFunction<trocDecGetDecodeStatus>(g_rocdecode_handle, "rocDecGetDecodeStatus");
  dl_rocDecReconfigureDecoder =
      bindFunction<trocDecReconfigureDecoder>(g_rocdecode_handle, "rocDecReconfigureDecoder");
  dl_rocDecGetVideoFrame =
      bindFunction<trocDecGetVideoFrame>(g_rocdecode_handle, "rocDecGetVideoFrame");
  dl_rocDecGetErrorName =
      bindFunction<trocDecGetErrorName>(g_rocdecode_handle, "rocDecGetErrorName");

  // Load parser function pointers
  dl_rocDecCreateVideoParser =
      bindFunction<trocDecCreateVideoParser>(g_rocparser_handle, "rocDecCreateVideoParser");
  dl_rocDecParseVideoData =
      bindFunction<trocDecParseVideoData>(g_rocparser_handle, "rocDecParseVideoData");
  dl_rocDecDestroyVideoParser =
      bindFunction<trocDecDestroyVideoParser>(g_rocparser_handle, "rocDecDestroyVideoParser");

  return isLoaded();
}

} // namespace facebook::torchcodec

// Actual function definitions that forward to the dynamically loaded pointers.
// These are compiled against and called by the RocDecode device interface code.
extern "C" {

rocDecStatus ROCDECAPI
rocDecCreateDecoder(rocDecDecoderHandle* decoder_handle,
                    RocDecoderCreateInfo* decoder_create_info) {
  TORCH_CHECK(
      facebook::torchcodec::dl_rocDecCreateDecoder,
      "rocDecCreateDecoder called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecCreateDecoder(
      decoder_handle, decoder_create_info);
}

rocDecStatus ROCDECAPI
rocDecDestroyDecoder(rocDecDecoderHandle decoder_handle) {
  TORCH_CHECK(
      facebook::torchcodec::dl_rocDecDestroyDecoder,
      "rocDecDestroyDecoder called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecDestroyDecoder(decoder_handle);
}

rocDecStatus ROCDECAPI rocDecGetDecoderCaps(RocdecDecodeCaps* pdc) {
  TORCH_CHECK(
      facebook::torchcodec::dl_rocDecGetDecoderCaps,
      "rocDecGetDecoderCaps called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecGetDecoderCaps(pdc);
}

rocDecStatus ROCDECAPI
rocDecDecodeFrame(rocDecDecoderHandle decoder_handle,
                  RocdecPicParams* pic_params) {
  TORCH_CHECK(
      facebook::torchcodec::dl_rocDecDecodeFrame,
      "rocDecDecodeFrame called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecDecodeFrame(decoder_handle, pic_params);
}

rocDecStatus ROCDECAPI
rocDecGetDecodeStatus(rocDecDecoderHandle decoder_handle, int pic_idx,
                      RocdecDecodeStatus* decode_status) {
  TORCH_CHECK(
      facebook::torchcodec::dl_rocDecGetDecodeStatus,
      "rocDecGetDecodeStatus called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecGetDecodeStatus(
      decoder_handle, pic_idx, decode_status);
}

rocDecStatus ROCDECAPI
rocDecReconfigureDecoder(rocDecDecoderHandle decoder_handle,
                         RocdecReconfigureDecoderInfo* reconfig_params) {
  TORCH_CHECK(
      facebook::torchcodec::dl_rocDecReconfigureDecoder,
      "rocDecReconfigureDecoder called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecReconfigureDecoder(
      decoder_handle, reconfig_params);
}

rocDecStatus ROCDECAPI
rocDecGetVideoFrame(rocDecDecoderHandle decoder_handle, int pic_idx,
                    void* dev_mem_ptr[3], uint32_t* horizontal_pitch,
                    RocdecProcParams* vid_postproc_params) {
  TORCH_CHECK(
      facebook::torchcodec::dl_rocDecGetVideoFrame,
      "rocDecGetVideoFrame called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecGetVideoFrame(
      decoder_handle, pic_idx, dev_mem_ptr, horizontal_pitch,
      vid_postproc_params);
}

const char* ROCDECAPI rocDecGetErrorName(rocDecStatus rocdec_status) {
  TORCH_CHECK(
      facebook::torchcodec::dl_rocDecGetErrorName,
      "rocDecGetErrorName called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecGetErrorName(rocdec_status);
}

rocDecStatus ROCDECAPI
rocDecCreateVideoParser(RocdecVideoParser* parser_handle,
                        RocdecParserParams* params) {
  TORCH_CHECK(
      facebook::torchcodec::dl_rocDecCreateVideoParser,
      "rocDecCreateVideoParser called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecCreateVideoParser(
      parser_handle, params);
}

rocDecStatus ROCDECAPI
rocDecParseVideoData(RocdecVideoParser parser_handle,
                     RocdecSourceDataPacket* packet) {
  TORCH_CHECK(
      facebook::torchcodec::dl_rocDecParseVideoData,
      "rocDecParseVideoData called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecParseVideoData(parser_handle, packet);
}

rocDecStatus ROCDECAPI
rocDecDestroyVideoParser(RocdecVideoParser parser_handle) {
  TORCH_CHECK(
      facebook::torchcodec::dl_rocDecDestroyVideoParser,
      "rocDecDestroyVideoParser called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecDestroyVideoParser(parser_handle);
}

} // extern "C"
