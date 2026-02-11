// Vendored rocDecode API header for TorchCodec ROCm backend.
// This is a minimal subset of the rocDecode public API sufficient for
// TorchCodec's needs. It allows compilation without requiring rocDecode
// headers to be installed at build time; the actual library is loaded
// at runtime via dlopen().
//
// Original source: https://github.com/ROCm/rocm-systems/projects/rocdecode
// Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// MIT License

#pragma once

#include <cstdint>

#ifndef ROCDECAPI
#if defined(_WIN32)
#define ROCDECAPI __stdcall
#else
#define ROCDECAPI
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

// ---- Handles ----
typedef void* rocDecDecoderHandle;

// ---- Status codes ----
typedef enum rocDecStatus_enum {
  ROCDEC_DEVICE_INVALID = -1,
  ROCDEC_CONTEXT_INVALID = -2,
  ROCDEC_RUNTIME_ERROR = -3,
  ROCDEC_OUTOF_MEMORY = -4,
  ROCDEC_INVALID_PARAMETER = -5,
  ROCDEC_NOT_IMPLEMENTED = -6,
  ROCDEC_NOT_INITIALIZED = -7,
  ROCDEC_NOT_SUPPORTED = -8,
  ROCDEC_SUCCESS = 0,
} rocDecStatus;

// ---- Codec types ----
typedef enum rocDecVideoCodec_enum {
  rocDecVideoCodec_MPEG1 = 0,
  rocDecVideoCodec_MPEG2,
  rocDecVideoCodec_MPEG4,
  rocDecVideoCodec_AVC,
  rocDecVideoCodec_HEVC,
  rocDecVideoCodec_AV1,
  rocDecVideoCodec_VP8,
  rocDecVideoCodec_VP9,
  rocDecVideoCodec_JPEG,
  rocDecVideoCodec_NumCodecs,
  rocDecVideoCodec_YUV420 =
      (('I' << 24) | ('Y' << 16) | ('U' << 8) | ('V')),
  rocDecVideoCodec_YV12 =
      (('Y' << 24) | ('V' << 16) | ('1' << 8) | ('2')),
  rocDecVideoCodec_NV12 =
      (('N' << 24) | ('V' << 16) | ('1' << 8) | ('2')),
  rocDecVideoCodec_YUYV =
      (('Y' << 24) | ('U' << 16) | ('Y' << 8) | ('V')),
  rocDecVideoCodec_UYVY =
      (('U' << 24) | ('Y' << 16) | ('V' << 8) | ('Y'))
} rocDecVideoCodec;

// ---- Surface formats ----
typedef enum rocDecVideoSurfaceFormat_enum {
  rocDecVideoSurfaceFormat_NV12 = 0,
  rocDecVideoSurfaceFormat_P016 = 1,
  rocDecVideoSurfaceFormat_YUV444 = 2,
  rocDecVideoSurfaceFormat_YUV444_16Bit = 3,
  rocDecVideoSurfaceFormat_YUV420 = 4,
  rocDecVideoSurfaceFormat_YUV420_16Bit = 5,
  rocDecVideoSurfaceFormat_YUV422 = 6,
  rocDecVideoSurfaceFormat_YUV422_16Bit = 7,
} rocDecVideoSurfaceFormat;

// ---- Chroma formats ----
typedef enum rocDecVideoChromaFormat_enum {
  rocDecVideoChromaFormat_Monochrome = 0,
  rocDecVideoChromaFormat_420,
  rocDecVideoChromaFormat_422,
  rocDecVideoChromaFormat_444
} rocDecVideoChromaFormat;

// ---- Decode status ----
typedef enum rocDecodeStatus_enum {
  rocDecodeStatus_Invalid = 0,
  rocDecodeStatus_InProgress = 1,
  rocDecodeStatus_Success = 2,
  rocDecodeStatus_Error = 8,
  rocDecodeStatus_Error_Concealed = 9,
  rocDecodeStatus_Displaying = 10,
} rocDecDecodeStatus;

// ---- Decode caps ----
typedef struct {
  rocDecVideoCodec codec_type;
  rocDecVideoChromaFormat chroma_format;
  uint32_t bit_depth_minus_8;
  uint32_t reserved_1[3];
  uint8_t is_supported;
  uint8_t num_nalu_per_chunk;
  uint16_t output_format_mask;
  uint32_t max_width;
  uint32_t max_height;
  uint32_t max_mb_count;
  uint16_t min_width;
  uint16_t min_height;
  uint32_t reserved_2[11];
} RocdecDecodeCaps;

// ---- Decoder create info ----
typedef struct {
  uint32_t codec_type;        // rocDecVideoCodec
  uint32_t chroma_format;     // rocDecVideoChromaFormat
  uint32_t bit_depth_minus_8;
  uint32_t output_format;     // rocDecVideoSurfaceFormat
  uint32_t internal_decode_flag;
  uint32_t width;
  uint32_t height;
  uint32_t max_width;
  uint32_t max_height;
  uint32_t num_decode_surfaces;
  uint32_t num_output_surfaces;
  uint32_t target_width;
  uint32_t target_height;
  struct {
    int16_t left;
    int16_t top;
    int16_t right;
    int16_t bottom;
  } target_rect;
  struct {
    int16_t left;
    int16_t top;
    int16_t right;
    int16_t bottom;
  } display_rect;
  int device_id;
  uint32_t reserved[4];
} RocDecoderCreateInfo;

// ---- Pic params (opaque - passed through from parser callbacks) ----
typedef struct _RocdecPicParams {
  int pic_width;
  int pic_height;
  int curr_pic_idx;
  int field_pic_flag;
  int bottom_field_flag;
  int second_field;
  int bitstream_data_len;
  const uint8_t* bitstream_data;
  int num_slices;
  uint32_t ref_pic_flag;
  int intra_pic_flag;
  uint32_t reserved[30];
  // Codec-specific union follows but we treat it as opaque
  uint8_t codec_data[4096];
} RocdecPicParams;

// ---- Process params for GetVideoFrame ----
typedef struct {
  int progressive_frame;
  int second_field;
  int top_field_first;
  int unpaired_field;
  uint32_t reserved_flags;
  uint32_t reserved_zero;
  uint64_t raw_input_dptr;
  uint32_t raw_input_pitch;
  uint32_t raw_input_format;
  uint32_t raw_output_dptr;
  uint32_t raw_output_pitch;
  uint32_t reserved_1;
  void* output_stream;
  uint32_t reserved[8];
} RocdecProcParams;

// ---- Decode status query ----
typedef struct {
  rocDecDecodeStatus decode_status;
  uint32_t reserved[31];
} RocdecDecodeStatus;

// ---- Reconfigure info ----
typedef struct {
  uint32_t width;
  uint32_t height;
  uint32_t target_width;
  uint32_t target_height;
  uint32_t num_decode_surfaces;
  uint32_t reserved[12];
  struct {
    int16_t left;
    int16_t top;
    int16_t right;
    int16_t bottom;
  } target_rect;
  struct {
    int16_t left;
    int16_t top;
    int16_t right;
    int16_t bottom;
  } display_rect;
} RocdecReconfigureDecoderInfo;

// ---- API functions ----
// These are the functions we dynamically load at runtime.
// Declarations here serve as documentation; actual binding
// happens in RocDecodeRuntimeLoader.cpp

extern rocDecStatus ROCDECAPI
rocDecCreateDecoder(rocDecDecoderHandle* decoder_handle,
                    RocDecoderCreateInfo* decoder_create_info);

extern rocDecStatus ROCDECAPI
rocDecDestroyDecoder(rocDecDecoderHandle decoder_handle);

extern rocDecStatus ROCDECAPI
rocDecGetDecoderCaps(RocdecDecodeCaps* pdc);

extern rocDecStatus ROCDECAPI
rocDecDecodeFrame(rocDecDecoderHandle decoder_handle,
                  RocdecPicParams* pic_params);

extern rocDecStatus ROCDECAPI
rocDecGetDecodeStatus(rocDecDecoderHandle decoder_handle, int pic_idx,
                      RocdecDecodeStatus* decode_status);

extern rocDecStatus ROCDECAPI
rocDecReconfigureDecoder(rocDecDecoderHandle decoder_handle,
                         RocdecReconfigureDecoderInfo* reconfig_params);

extern rocDecStatus ROCDECAPI
rocDecGetVideoFrame(rocDecDecoderHandle decoder_handle, int pic_idx,
                    void* dev_mem_ptr[3], uint32_t* horizontal_pitch,
                    RocdecProcParams* vid_postproc_params);

extern const char* ROCDECAPI rocDecGetErrorName(rocDecStatus rocdec_status);

#if defined(__cplusplus)
}
#endif
