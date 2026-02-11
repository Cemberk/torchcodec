// Vendored rocParser API header for TorchCodec ROCm backend.
// Minimal subset of the rocDecode parser API.
//
// Original source: https://github.com/ROCm/rocm-systems/projects/rocdecode
// Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// MIT License

#pragma once

#include "rocdecode.h"

#if defined(__cplusplus)
extern "C" {
#endif

// ---- Parser handle ----
typedef void* RocdecVideoParser;
typedef uint64_t RocdecTimeStamp;

// ---- Video format (from parser sequence callback) ----
typedef struct {
  rocDecVideoCodec codec;
  struct {
    uint32_t numerator;
    uint32_t denominator;
  } frame_rate;
  uint8_t progressive_sequence;
  uint8_t bit_depth_luma_minus8;
  uint8_t bit_depth_chroma_minus8;
  uint8_t min_num_decode_surfaces;
  uint32_t coded_width;
  uint32_t coded_height;
  struct {
    int left;
    int top;
    int right;
    int bottom;
  } display_area;
  rocDecVideoChromaFormat chroma_format;
  uint32_t bitrate;
  struct {
    int x;
    int y;
  } display_aspect_ratio;
  struct {
    uint8_t video_format : 3;
    uint8_t video_full_range_flag : 1;
    uint8_t reserved_zero_bits : 4;
    uint8_t color_primaries;
    uint8_t transfer_characteristics;
    uint8_t matrix_coefficients;
  } video_signal_description;
  uint32_t seqhdr_data_length;
  uint32_t reconfig_options;
} RocdecVideoFormat;

// ---- Extended video format ----
typedef struct {
  RocdecVideoFormat format;
  uint32_t max_width;
  uint32_t max_height;
  uint8_t raw_seqhdr_data[1024];
} RocdecVideoFormatEx;

// ---- Packet flags ----
typedef enum {
  ROCDEC_PKT_ENDOFSTREAM = 0x01,
  ROCDEC_PKT_TIMESTAMP = 0x02,
  ROCDEC_PKT_DISCONTINUITY = 0x04,
  ROCDEC_PKT_ENDOFPICTURE = 0x08,
  ROCDEC_PKT_NOTIFY_EOS = 0x10,
} RocdecVideoPacketFlags;

// ---- Source data packet ----
typedef struct _RocdecSourceDataPacket {
  uint32_t flags;
  uint32_t payload_size;
  const uint8_t* payload;
  RocdecTimeStamp pts;
} RocdecSourceDataPacket;

// ---- Display info (from parser display callback) ----
typedef struct _RocdecParserDispInfo {
  int picture_index;
  int progressive_frame;
  int top_field_first;
  int repeat_first_field;
  RocdecTimeStamp pts;
} RocdecParserDispInfo;

// ---- SEI message structures ----
typedef struct _RocdecSeiMessage {
  uint8_t sei_message_type;
  uint8_t reserved[3];
  uint32_t sei_message_size;
} RocdecSeiMessage;

typedef struct _RocdecSeiMessageInfo {
  void* sei_data;
  RocdecSeiMessage* sei_message;
  uint32_t sei_message_count;
  uint32_t picIdx;
} RocdecSeiMessageInfo;

// ---- Parser callback function pointer types ----
typedef int(ROCDECAPI* PFNVIDSEQUENCECALLBACK)(void*, RocdecVideoFormat*);
typedef int(ROCDECAPI* PFNVIDDECODECALLBACK)(void*, RocdecPicParams*);
typedef int(ROCDECAPI* PFNVIDDISPLAYCALLBACK)(void*, RocdecParserDispInfo*);
typedef int(ROCDECAPI* PFNVIDSEIMSGCALLBACK)(void*, RocdecSeiMessageInfo*);

// ---- Parser params ----
typedef struct _RocdecParserParams {
  rocDecVideoCodec codec_type;
  uint32_t max_num_decode_surfaces;
  uint32_t clock_rate;
  uint32_t error_threshold;
  uint32_t max_display_delay;
  uint32_t annex_b : 1;
  uint32_t reserved : 31;
  uint32_t reserved_1[4];
  void* user_data;
  PFNVIDSEQUENCECALLBACK pfn_sequence_callback;
  PFNVIDDECODECALLBACK pfn_decode_picture;
  PFNVIDDISPLAYCALLBACK pfn_display_picture;
  PFNVIDSEIMSGCALLBACK pfn_get_sei_msg;
  void* reserved_2[5];
  RocdecVideoFormatEx* ext_video_info;
} RocdecParserParams;

// ---- Parser API functions ----
extern rocDecStatus ROCDECAPI
rocDecCreateVideoParser(RocdecVideoParser* parser_handle,
                        RocdecParserParams* params);

extern rocDecStatus ROCDECAPI
rocDecParseVideoData(RocdecVideoParser parser_handle,
                     RocdecSourceDataPacket* packet);

extern rocDecStatus ROCDECAPI
rocDecDestroyVideoParser(RocdecVideoParser parser_handle);

#if defined(__cplusplus)
}
#endif
