// Vendored rocDecode API header for TorchCodec ROCm backend.
//
// This is a copy of the rocDecode v1.5.0 public API header that allows
// compilation without requiring rocDecode headers to be installed at build
// time; the actual library is loaded at runtime via dlopen().
//
// IMPORTANT: The struct layouts in this file MUST exactly match the installed
// rocDecode library (ABI compatibility). If rocDecode is updated, this file
// must be updated to match. Struct layout mismatches cause GPU memory access
// faults and crashes.
//
// Source: rocDecode v1.5.0 (ROCm 7.2)
//   https://github.com/ROCm/rocDecode
//
// Copyright (c) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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

/*********************************************************************************/
// HANDLE of rocDecDecoder
// Used in subsequent API calls after rocDecCreateDecoder
/*********************************************************************************/
typedef void* rocDecDecoderHandle;

/*********************************************************************************/
// rocDecoder return status enums
/*********************************************************************************/
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

/*********************************************************************************/
// Video codec enums
/*********************************************************************************/
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

/*********************************************************************************/
// Video surface format enums used for output format of decoded output
/*********************************************************************************/
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

/*********************************************************************************/
// Chroma format enums
/*********************************************************************************/
typedef enum rocDecVideoChromaFormat_enum {
  rocDecVideoChromaFormat_Monochrome = 0,
  rocDecVideoChromaFormat_420,
  rocDecVideoChromaFormat_422,
  rocDecVideoChromaFormat_444
} rocDecVideoChromaFormat;

/*********************************************************************************/
// Decode status enums
/*********************************************************************************/
typedef enum rocDecodeStatus_enum {
  rocDecodeStatus_Invalid = 0,
  rocDecodeStatus_InProgress = 1,
  rocDecodeStatus_Success = 2,
  // 3 to 7 reserved for future use
  rocDecodeStatus_Error = 8,
  rocDecodeStatus_Error_Concealed = 9,
  rocDecodeStatus_Displaying = 10,
} rocDecDecodeStatus;

/*********************************************************************************/
// RocdecDecodeCaps - used in rocDecGetDecoderCaps API
/*********************************************************************************/
typedef struct _RocdecDecodeCaps {
  uint8_t device_id;                     /**< IN: device id (0 for first, 1 for second, etc.) */
  rocDecVideoCodec codec_type;           /**< IN: rocDecVideoCodec_XXX */
  rocDecVideoChromaFormat chroma_format; /**< IN: rocDecVideoChromaFormat_XXX */
  uint32_t bit_depth_minus_8;            /**< IN: The Value "BitDepth minus 8" */
  uint32_t reserved_1[3];                /**< Reserved for future use - set to zero */
  uint8_t is_supported;                  /**< OUT: 1 if codec supported, 0 if not */
  uint8_t num_decoders;                  /**< OUT: Number of decoders supporting IN params */
  uint16_t output_format_mask;           /**< OUT: each bit represents rocDecVideoSurfaceFormat enum */
  uint32_t max_width;                    /**< OUT: Max supported coded width in pixels */
  uint32_t max_height;                   /**< OUT: Max supported coded height in pixels */
  uint16_t min_width;                    /**< OUT: Min supported coded width in pixels */
  uint16_t min_height;                   /**< OUT: Min supported coded height in pixels */
  uint32_t reserved_2[6];               /**< Reserved for future use - set to zero */
} RocdecDecodeCaps;

/*********************************************************************************/
// RocDecoderCreateInfo - used in rocDecCreateDecoder API
/*********************************************************************************/
typedef struct _RocDecoderCreateInfo {
  uint8_t device_id;                     /**< IN: device id (0 for first, 1 for second, etc.) */
  uint32_t width;                        /**< IN: Coded sequence width in pixels */
  uint32_t height;                       /**< IN: Coded sequence height in pixels */
  uint32_t num_decode_surfaces;          /**< IN: Maximum number of internal decode surfaces */
  rocDecVideoCodec codec_type;           /**< IN: rocDecVideoCodec_XXX */
  rocDecVideoChromaFormat chroma_format; /**< IN: rocDecVideoChromaFormat_XXX */
  uint32_t bit_depth_minus_8;            /**< IN: The value "BitDepth minus 8" */
  uint32_t intra_decode_only;            /**< IN: Set 1 only if video has all intra frames */
  uint32_t max_width;                    /**< IN: Coded sequence max width for reconfigure */
  uint32_t max_height;                   /**< IN: Coded sequence max height for reconfigure */
  struct {
    int16_t left;
    int16_t top;
    int16_t right;
    int16_t bottom;
  } display_rect;                         /**< IN: area of the frame to display */
  rocDecVideoSurfaceFormat output_format; /**< IN: rocDecVideoSurfaceFormat_XXX */
  uint32_t target_width;                  /**< IN: Post-processed output width (aligned to 2) */
  uint32_t target_height;                 /**< IN: Post-processed output height (aligned to 2) */
  uint32_t num_output_surfaces;           /**< IN: Max number of output surfaces mapped */
  struct {
    int16_t left;
    int16_t top;
    int16_t right;
    int16_t bottom;
  } target_rect;          /**< IN: target rectangle in output frame */
  uint32_t reserved_2[4]; /**< Reserved for future use - set to zero */
} RocDecoderCreateInfo;

/*********************************************************************************/
// RocdecDecodeStatus - used in rocDecGetDecodeStatus API
/*********************************************************************************/
typedef struct _RocdecDecodeStatus {
  rocDecDecodeStatus decode_status;
  uint32_t reserved[31];
  void* p_reserved[8];
} RocdecDecodeStatus;

/*********************************************************************************/
// RocdecReconfigureDecoderInfo - used in rocDecReconfigureDecoder API
/*********************************************************************************/
typedef struct _RocdecReconfigureDecoderInfo {
  uint32_t width;               /**< IN: Coded sequence width, MUST be <= max_width */
  uint32_t height;              /**< IN: Coded sequence height, MUST be <= max_height */
  uint32_t target_width;        /**< IN: Post processed output width */
  uint32_t target_height;       /**< IN: Post processed output height */
  uint32_t num_decode_surfaces; /**< IN: Maximum number of internal decode surfaces */
  uint32_t bit_depth_minus_8;   /**< IN: The Value "BitDepth minus 8" */
  uint32_t reserved_1[11];      /**< Reserved for future use. Set to zero */
  struct {
    int16_t left;
    int16_t top;
    int16_t right;
    int16_t bottom;
  } display_rect; /**< IN: area of the frame to display */
  struct {
    int16_t left;
    int16_t top;
    int16_t right;
    int16_t bottom;
  } target_rect;           /**< IN: target rectangle in output frame */
  uint32_t reserved_2[11]; /**< Reserved for future use. Set to zero */
} RocdecReconfigureDecoderInfo;

/*********************************************************************************/
// AVC/H.264 Picture Entry
/*********************************************************************************/
typedef struct _RocdecAvcPicture {
  int pic_idx;
  uint32_t frame_idx;
  uint32_t flags;
  int32_t top_field_order_cnt;
  int32_t bottom_field_order_cnt;
  uint32_t reserved[4];
} RocdecAvcPicture;

#define RocdecAvcPicture_FLAGS_INVALID 0x00000001
#define RocdecAvcPicture_FLAGS_TOP_FIELD 0x00000002
#define RocdecAvcPicture_FLAGS_BOTTOM_FIELD 0x00000004
#define RocdecAvcPicture_FLAGS_SHORT_TERM_REFERENCE 0x00000008
#define RocdecAvcPicture_FLAGS_LONG_TERM_REFERENCE 0x00000010
#define RocdecAvcPicture_FLAGS_NON_EXISTING 0x00000020

/*********************************************************************************/
// HEVC Picture Entry
/*********************************************************************************/
typedef struct _RocdecHevcPicture {
  int pic_idx;
  int poc;
  uint32_t flags;
  uint32_t reserved[4];
} RocdecHevcPicture;

#define RocdecHevcPicture_INVALID 0x00000001
#define RocdecHevcPicture_FIELD_PIC 0x00000002
#define RocdecHevcPicture_BOTTOM_FIELD 0x00000004
#define RocdecHevcPicture_LONG_TERM_REFERENCE 0x00000008
#define RocdecHevcPicture_RPS_ST_CURR_BEFORE 0x00000010
#define RocdecHevcPicture_RPS_ST_CURR_AFTER 0x00000020
#define RocdecHevcPicture_RPS_LT_CURR 0x00000040

/*********************************************************************************/
// JPEG picture parameters (placeholder)
/*********************************************************************************/
typedef struct _RocdecJPEGPicParams {
  int reserved;
} RocdecJPEGPicParams;

/*********************************************************************************/
// MPEG2 QMatrix
/*********************************************************************************/
typedef struct _RocdecMpeg2QMatrix {
  int32_t load_intra_quantiser_matrix;
  int32_t load_non_intra_quantiser_matrix;
  int32_t load_chroma_intra_quantiser_matrix;
  int32_t load_chroma_non_intra_quantiser_matrix;
  uint8_t intra_quantiser_matrix[64];
  uint8_t non_intra_quantiser_matrix[64];
  uint8_t chroma_intra_quantiser_matrix[64];
  uint8_t chroma_non_intra_quantiser_matrix[64];
} RocdecMpeg2QMatrix;

/*********************************************************************************/
// MPEG2 picture parameters
/*********************************************************************************/
typedef struct _RocdecMpeg2PicParams {
  uint16_t horizontal_size;
  uint16_t vertical_size;
  uint32_t forward_reference_pic;
  uint32_t backward_reference_picture;
  int32_t picture_coding_type;
  int32_t f_code;
  union {
    struct {
      uint32_t intra_dc_precision : 2;
      uint32_t picture_structure : 2;
      uint32_t top_field_first : 1;
      uint32_t frame_pred_frame_dct : 1;
      uint32_t concealment_motion_vectors : 1;
      uint32_t q_scale_type : 1;
      uint32_t intra_vlc_format : 1;
      uint32_t alternate_scan : 1;
      uint32_t repeat_first_field : 1;
      uint32_t progressive_frame : 1;
      uint32_t is_first_field : 1;
    } bits;
    uint32_t value;
  } picture_coding_extension;
  RocdecMpeg2QMatrix q_matrix;
  uint32_t reserved[4];
} RocdecMpeg2PicParams;

/*********************************************************************************/
// VC1 picture parameters (placeholder)
/*********************************************************************************/
typedef struct _RocdecVc1PicParams {
  int reserved;
} RocdecVc1PicParams;

/*********************************************************************************/
// AVC picture parameters (VA-API compatible)
/*********************************************************************************/
typedef struct _RocdecAvcPicParams {
  RocdecAvcPicture curr_pic;
  RocdecAvcPicture ref_frames[16];
  uint16_t picture_width_in_mbs_minus1;
  uint16_t picture_height_in_mbs_minus1;
  uint8_t bit_depth_luma_minus8;
  uint8_t bit_depth_chroma_minus8;
  uint8_t num_ref_frames;
  union {
    struct {
      uint32_t chroma_format_idc : 2;
      uint32_t residual_colour_transform_flag : 1;
      uint32_t gaps_in_frame_num_value_allowed_flag : 1;
      uint32_t frame_mbs_only_flag : 1;
      uint32_t mb_adaptive_frame_field_flag : 1;
      uint32_t direct_8x8_inference_flag : 1;
      uint32_t MinLumaBiPredSize8x8 : 1;
      uint32_t log2_max_frame_num_minus4 : 4;
      uint32_t pic_order_cnt_type : 2;
      uint32_t log2_max_pic_order_cnt_lsb_minus4 : 4;
      uint32_t delta_pic_order_always_zero_flag : 1;
    } bits;
    uint32_t value;
  } seq_fields;
  uint8_t num_slice_groups_minus1;
  uint8_t slice_group_map_type;
  uint16_t slice_group_change_rate_minus1;
  int8_t pic_init_qp_minus26;
  int8_t pic_init_qs_minus26;
  int8_t chroma_qp_index_offset;
  int8_t second_chroma_qp_index_offset;
  union {
    struct {
      uint32_t entropy_coding_mode_flag : 1;
      uint32_t weighted_pred_flag : 1;
      uint32_t weighted_bipred_idc : 2;
      uint32_t transform_8x8_mode_flag : 1;
      uint32_t field_pic_flag : 1;
      uint32_t constrained_intra_pred_flag : 1;
      uint32_t pic_order_present_flag : 1;
      uint32_t deblocking_filter_control_present_flag : 1;
      uint32_t redundant_pic_cnt_present_flag : 1;
      uint32_t reference_pic_flag : 1;
    } bits;
    uint32_t value;
  } pic_fields;
  uint16_t frame_num;
  uint32_t reserved[8];
} RocdecAvcPicParams;

/*********************************************************************************/
// AVC slice parameters (VA-API compatible)
/*********************************************************************************/
typedef struct _RocdecAvcSliceParams {
  uint32_t slice_data_size;
  uint32_t slice_data_offset;
  uint32_t slice_data_flag;
  uint16_t slice_data_bit_offset;
  uint16_t first_mb_in_slice;
  uint8_t slice_type;
  uint8_t direct_spatial_mv_pred_flag;
  uint8_t num_ref_idx_l0_active_minus1;
  uint8_t num_ref_idx_l1_active_minus1;
  uint8_t cabac_init_idc;
  int8_t slice_qp_delta;
  uint8_t disable_deblocking_filter_idc;
  int8_t slice_alpha_c0_offset_div2;
  int8_t slice_beta_offset_div2;
  RocdecAvcPicture ref_pic_list_0[32];
  RocdecAvcPicture ref_pic_list_1[32];
  uint8_t luma_log2_weight_denom;
  uint8_t chroma_log2_weight_denom;
  uint8_t luma_weight_l0_flag;
  int16_t luma_weight_l0[32];
  int16_t luma_offset_l0[32];
  uint8_t chroma_weight_l0_flag;
  int16_t chroma_weight_l0[32][2];
  int16_t chroma_offset_l0[32][2];
  uint8_t luma_weight_l1_flag;
  int16_t luma_weight_l1[32];
  int16_t luma_offset_l1[32];
  uint8_t chroma_weight_l1_flag;
  int16_t chroma_weight_l1[32][2];
  int16_t chroma_offset_l1[32][2];
  uint32_t reserved[4];
} RocdecAvcSliceParams;

/*********************************************************************************/
// AVC Inverse Quantization Matrix (VA-API compatible)
/*********************************************************************************/
typedef struct _RocdecAvcIQMatrix {
  uint8_t scaling_list_4x4[6][16];
  uint8_t scaling_list_8x8[2][64];
  uint32_t reserved[4];
} RocdecAvcIQMatrix;

/*********************************************************************************/
// HEVC picture parameters
/*********************************************************************************/
typedef struct _RocdecHevcPicParams {
  RocdecHevcPicture curr_pic;
  RocdecHevcPicture ref_frames[15];
  uint16_t picture_width_in_luma_samples;
  uint16_t picture_height_in_luma_samples;
  union {
    struct {
      uint32_t chroma_format_idc : 2;
      uint32_t separate_colour_plane_flag : 1;
      uint32_t pcm_enabled_flag : 1;
      uint32_t scaling_list_enabled_flag : 1;
      uint32_t transform_skip_enabled_flag : 1;
      uint32_t amp_enabled_flag : 1;
      uint32_t strong_intra_smoothing_enabled_flag : 1;
      uint32_t sign_data_hiding_enabled_flag : 1;
      uint32_t constrained_intra_pred_flag : 1;
      uint32_t cu_qp_delta_enabled_flag : 1;
      uint32_t weighted_pred_flag : 1;
      uint32_t weighted_bipred_flag : 1;
      uint32_t transquant_bypass_enabled_flag : 1;
      uint32_t tiles_enabled_flag : 1;
      uint32_t entropy_coding_sync_enabled_flag : 1;
      uint32_t pps_loop_filter_across_slices_enabled_flag : 1;
      uint32_t loop_filter_across_tiles_enabled_flag : 1;
      uint32_t pcm_loop_filter_disabled_flag : 1;
      uint32_t no_pic_reordering_flag : 1;
      uint32_t no_bi_pred_flag : 1;
      uint32_t reserved_bits : 11;
    } bits;
    uint32_t value;
  } pic_fields;
  uint8_t sps_max_dec_pic_buffering_minus1;
  uint8_t bit_depth_luma_minus8;
  uint8_t bit_depth_chroma_minus8;
  uint8_t pcm_sample_bit_depth_luma_minus1;
  uint8_t pcm_sample_bit_depth_chroma_minus1;
  uint8_t log2_min_luma_coding_block_size_minus3;
  uint8_t log2_diff_max_min_luma_coding_block_size;
  uint8_t log2_min_luma_transform_block_size_minus2;
  uint8_t log2_diff_max_min_luma_transform_block_size;
  uint8_t log2_min_pcm_luma_coding_block_size_minus3;
  uint8_t log2_diff_max_min_pcm_luma_coding_block_size;
  uint8_t max_transform_hierarchy_depth_intra;
  uint8_t max_transform_hierarchy_depth_inter;
  int8_t init_qp_minus26;
  uint8_t diff_cu_qp_delta_depth;
  int8_t pps_cb_qp_offset;
  int8_t pps_cr_qp_offset;
  uint8_t log2_parallel_merge_level_minus2;
  uint8_t num_tile_columns_minus1;
  uint8_t num_tile_rows_minus1;
  uint16_t column_width_minus1[19];
  uint16_t row_height_minus1[21];
  union {
    struct {
      uint32_t lists_modification_present_flag : 1;
      uint32_t long_term_ref_pics_present_flag : 1;
      uint32_t sps_temporal_mvp_enabled_flag : 1;
      uint32_t cabac_init_present_flag : 1;
      uint32_t output_flag_present_flag : 1;
      uint32_t dependent_slice_segments_enabled_flag : 1;
      uint32_t pps_slice_chroma_qp_offsets_present_flag : 1;
      uint32_t sample_adaptive_offset_enabled_flag : 1;
      uint32_t deblocking_filter_override_enabled_flag : 1;
      uint32_t pps_disable_deblocking_filter_flag : 1;
      uint32_t slice_segment_header_extension_present_flag : 1;
      uint32_t rap_pic_flag : 1;
      uint32_t idr_pic_flag : 1;
      uint32_t intra_pic_flag : 1;
      uint32_t reserved_bits : 18;
    } bits;
    uint32_t value;
  } slice_parsing_fields;
  uint8_t log2_max_pic_order_cnt_lsb_minus4;
  uint8_t num_short_term_ref_pic_sets;
  uint8_t num_long_term_ref_pic_sps;
  uint8_t num_ref_idx_l0_default_active_minus1;
  uint8_t num_ref_idx_l1_default_active_minus1;
  int8_t pps_beta_offset_div2;
  int8_t pps_tc_offset_div2;
  uint8_t num_extra_slice_header_bits;
  uint32_t st_rps_bits;
  uint32_t reserved[8];
} RocdecHevcPicParams;

/*********************************************************************************/
// HEVC slice parameters
/*********************************************************************************/
typedef struct _RocdecHevcSliceParams {
  uint32_t slice_data_size;
  uint32_t slice_data_offset;
  uint32_t slice_data_flag;
  uint32_t slice_data_byte_offset;
  uint32_t slice_segment_address;
  uint8_t ref_pic_list[2][15];
  union {
    uint32_t value;
    struct {
      uint32_t last_slice_of_pic : 1;
      uint32_t dependent_slice_segment_flag : 1;
      uint32_t slice_type : 2;
      uint32_t color_plane_id : 2;
      uint32_t slice_sao_luma_flag : 1;
      uint32_t slice_sao_chroma_flag : 1;
      uint32_t mvd_l1_zero_flag : 1;
      uint32_t cabac_init_flag : 1;
      uint32_t slice_temporal_mvp_enabled_flag : 1;
      uint32_t slice_deblocking_filter_disabled_flag : 1;
      uint32_t collocated_from_l0_flag : 1;
      uint32_t slice_loop_filter_across_slices_enabled_flag : 1;
      uint32_t reserved : 18;
    } fields;
  } long_slice_flags;
  uint8_t collocated_ref_idx;
  uint8_t num_ref_idx_l0_active_minus1;
  uint8_t num_ref_idx_l1_active_minus1;
  int8_t slice_qp_delta;
  int8_t slice_cb_qp_offset;
  int8_t slice_cr_qp_offset;
  int8_t slice_beta_offset_div2;
  int8_t slice_tc_offset_div2;
  uint8_t luma_log2_weight_denom;
  int8_t delta_chroma_log2_weight_denom;
  int8_t delta_luma_weight_l0[15];
  int8_t luma_offset_l0[15];
  int8_t delta_chroma_weight_l0[15][2];
  int8_t chroma_offset_l0[15][2];
  int8_t delta_luma_weight_l1[15];
  int8_t luma_offset_l1[15];
  int8_t delta_chroma_weight_l1[15][2];
  int8_t chroma_offset_l1[15][2];
  uint8_t five_minus_max_num_merge_cand;
  uint16_t num_entry_point_offsets;
  uint16_t entry_offset_to_subset_array;
  uint16_t slice_data_num_emu_prevn_bytes;
  uint32_t reserved[2];
} RocdecHevcSliceParams;

/*********************************************************************************/
// HEVC IQ Matrix
/*********************************************************************************/
typedef struct _RocdecHevcIQMatrix {
  uint8_t scaling_list_4x4[6][16];
  uint8_t scaling_list_8x8[6][64];
  uint8_t scaling_list_16x16[6][64];
  uint8_t scaling_list_32x32[2][64];
  uint8_t scaling_list_dc_16x16[6];
  uint8_t scaling_list_dc_32x32[2];
  uint32_t reserved[4];
} RocdecHevcIQMatrix;

/*********************************************************************************/
// VP9 picture parameters (VA-API compatible)
/*********************************************************************************/
typedef struct _RocdecVp9PicParams {
  uint16_t frame_width;
  uint16_t frame_height;
  uint32_t reference_frames[8];
  union {
    struct {
      uint32_t subsampling_x : 1;
      uint32_t subsampling_y : 1;
      uint32_t frame_type : 1;
      uint32_t show_frame : 1;
      uint32_t error_resilient_mode : 1;
      uint32_t intra_only : 1;
      uint32_t allow_high_precision_mv : 1;
      uint32_t mcomp_filter_type : 3;
      uint32_t frame_parallel_decoding_mode : 1;
      uint32_t reset_frame_context : 2;
      uint32_t refresh_frame_context : 1;
      uint32_t frame_context_idx : 2;
      uint32_t segmentation_enabled : 1;
      uint32_t segmentation_temporal_update : 1;
      uint32_t segmentation_update_map : 1;
      uint32_t last_ref_frame : 3;
      uint32_t last_ref_frame_sign_bias : 1;
      uint32_t golden_ref_frame : 3;
      uint32_t golden_ref_frame_sign_bias : 1;
      uint32_t alt_ref_frame : 3;
      uint32_t alt_ref_frame_sign_bias : 1;
      uint32_t lossless_flag : 1;
    } bits;
    uint32_t value;
  } pic_fields;
  uint8_t filter_level;
  uint8_t sharpness_level;
  uint8_t log2_tile_rows;
  uint8_t log2_tile_columns;
  uint8_t frame_header_length_in_bytes;
  uint16_t first_partition_size;
  uint8_t mb_segment_tree_probs[7];
  uint8_t segment_pred_probs[3];
  uint8_t profile;
  uint8_t bit_depth;
  uint32_t va_reserved[8];
} RocdecVp9PicParams;

/*********************************************************************************/
// VP9 Segmentation Parameter
/*********************************************************************************/
typedef struct _RocdecVp9SegmentParameter {
  union {
    struct {
      uint16_t segment_reference_enabled : 1;
      uint16_t segment_reference : 2;
      uint16_t segment_reference_skipped : 1;
    } fields;
    uint16_t value;
  } segment_flags;
  uint8_t filter_level[4][2];
  int16_t luma_ac_quant_scale;
  int16_t luma_dc_quant_scale;
  int16_t chroma_ac_quant_scale;
  int16_t chroma_dc_quant_scale;
  uint32_t va_reserved[4];
} RocdecVp9SegmentParameter;

/*********************************************************************************/
// VP9 slice parameters (VA-API compatible)
/*********************************************************************************/
typedef struct _RocdecVp9SliceParams {
  uint32_t slice_data_size;
  uint32_t slice_data_offset;
  uint32_t slice_data_flag;
  RocdecVp9SegmentParameter seg_param[8];
  uint32_t va_reserved[4];
} RocdecVp9SliceParams;

/*********************************************************************************/
// AV1 Segmentation Information
/*********************************************************************************/
typedef struct _RocdecAv1SegmentationStruct {
  union {
    struct {
      uint32_t enabled : 1;
      uint32_t update_map : 1;
      uint32_t temporal_update : 1;
      uint32_t update_data : 1;
      uint32_t reserved : 28;
    } bits;
    uint32_t value;
  } segment_info_fields;
  int16_t feature_data[8][8];
  uint8_t feature_mask[8];
  uint32_t reserved[4];
} RocdecAv1SegmentationStruct;

/*********************************************************************************/
// AV1 Film Grain Information
/*********************************************************************************/
typedef struct _RocdecAv1FilmGrainStruct {
  union {
    struct {
      uint32_t apply_grain : 1;
      uint32_t chroma_scaling_from_luma : 1;
      uint32_t grain_scaling_minus_8 : 2;
      uint32_t ar_coeff_lag : 2;
      uint32_t ar_coeff_shift_minus_6 : 2;
      uint32_t grain_scale_shift : 2;
      uint32_t overlap_flag : 1;
      uint32_t clip_to_restricted_range : 1;
      uint32_t reserved : 20;
    } bits;
    uint32_t value;
  } film_grain_info_fields;
  uint16_t grain_seed;
  uint8_t num_y_points;
  uint8_t point_y_value[14];
  uint8_t point_y_scaling[14];
  uint8_t num_cb_points;
  uint8_t point_cb_value[10];
  uint8_t point_cb_scaling[10];
  uint8_t num_cr_points;
  uint8_t point_cr_value[10];
  uint8_t point_cr_scaling[10];
  int8_t ar_coeffs_y[24];
  int8_t ar_coeffs_cb[25];
  int8_t ar_coeffs_cr[25];
  uint8_t cb_mult;
  uint8_t cb_luma_mult;
  uint16_t cb_offset;
  uint8_t cr_mult;
  uint8_t cr_luma_mult;
  uint16_t cr_offset;
  uint32_t reserved[4];
} RocdecAv1FilmGrainStruct;

typedef enum {
  RocdecAv1TransformationIdentity = 0,
  RocdecAv1TransformationTranslation = 1,
  RocdecAv1TransformationRotzoom = 2,
  RocdecAv1TransformationAffine = 3,
  RocdecAv1TransformationCount
} RocdecAv1TransformationType;

typedef struct _RocdecAv1WarpedMotionParams {
  RocdecAv1TransformationType wmtype;
  int32_t wmmat[8];
  uint8_t invalid;
  uint32_t reserved[4];
} RocdecAv1WarpedMotionParams;

/*********************************************************************************/
// AV1 picture parameters
/*********************************************************************************/
typedef struct _RocdecAV1PicParams {
  uint8_t profile;
  uint8_t order_hint_bits_minus_1;
  uint8_t bit_depth_idx;
  uint8_t matrix_coefficients;
  union {
    struct {
      uint32_t still_picture : 1;
      uint32_t use_128x128_superblock : 1;
      uint32_t enable_filter_intra : 1;
      uint32_t enable_intra_edge_filter : 1;
      uint32_t enable_interintra_compound : 1;
      uint32_t enable_masked_compound : 1;
      uint32_t enable_dual_filter : 1;
      uint32_t enable_order_hint : 1;
      uint32_t enable_jnt_comp : 1;
      uint32_t enable_cdef : 1;
      uint32_t mono_chrome : 1;
      uint32_t color_range : 1;
      uint32_t subsampling_x : 1;
      uint32_t subsampling_y : 1;
      uint32_t chroma_sample_position : 1;
      uint32_t film_grain_params_present : 1;
      uint32_t reserved : 16;
    } fields;
    uint32_t value;
  } seq_info_fields;
  int current_frame;
  int current_display_picture;
  uint8_t anchor_frames_num;
  int* anchor_frames_list;
  uint16_t frame_width_minus1;
  uint16_t frame_height_minus1;
  uint16_t output_frame_width_in_tiles_minus_1;
  uint16_t output_frame_height_in_tiles_minus_1;
  int ref_frame_map[8];
  uint8_t ref_frame_idx[7];
  uint8_t primary_ref_frame;
  uint8_t order_hint;
  RocdecAv1SegmentationStruct seg_info;
  RocdecAv1FilmGrainStruct film_grain_info;
  uint8_t tile_cols;
  uint8_t tile_rows;
  uint16_t width_in_sbs_minus_1[63];
  uint16_t height_in_sbs_minus_1[63];
  uint16_t tile_count_minus_1;
  uint16_t context_update_tile_id;
  union {
    struct {
      uint32_t frame_type : 2;
      uint32_t show_frame : 1;
      uint32_t showable_frame : 1;
      uint32_t error_resilient_mode : 1;
      uint32_t disable_cdf_update : 1;
      uint32_t allow_screen_content_tools : 1;
      uint32_t force_integer_mv : 1;
      uint32_t allow_intrabc : 1;
      uint32_t use_superres : 1;
      uint32_t allow_high_precision_mv : 1;
      uint32_t is_motion_mode_switchable : 1;
      uint32_t use_ref_frame_mvs : 1;
      uint32_t disable_frame_end_update_cdf : 1;
      uint32_t uniform_tile_spacing_flag : 1;
      uint32_t allow_warped_motion : 1;
      uint32_t large_scale_tile : 1;
      uint32_t reserved : 15;
    } bits;
    uint32_t value;
  } pic_info_fields;
  uint8_t superres_scale_denominator;
  uint8_t interp_filter;
  uint8_t filter_level[2];
  uint8_t filter_level_u;
  uint8_t filter_level_v;
  union {
    struct {
      uint8_t sharpness_level : 3;
      uint8_t mode_ref_delta_enabled : 1;
      uint8_t mode_ref_delta_update : 1;
      uint8_t reserved : 3;
    } bits;
    uint8_t value;
  } loop_filter_info_fields;
  int8_t ref_deltas[8];
  int8_t mode_deltas[2];
  uint8_t base_qindex;
  int8_t y_dc_delta_q;
  int8_t u_dc_delta_q;
  int8_t u_ac_delta_q;
  int8_t v_dc_delta_q;
  int8_t v_ac_delta_q;
  union {
    struct {
      uint16_t using_qmatrix : 1;
      uint16_t qm_y : 4;
      uint16_t qm_u : 4;
      uint16_t qm_v : 4;
      uint16_t reserved : 3;
    } bits;
    uint16_t value;
  } qmatrix_fields;
  union {
    struct {
      uint32_t delta_q_present_flag : 1;
      uint32_t log2_delta_q_res : 2;
      uint32_t delta_lf_present_flag : 1;
      uint32_t log2_delta_lf_res : 2;
      uint32_t delta_lf_multi : 1;
      uint32_t tx_mode : 2;
      uint32_t reference_select : 1;
      uint32_t reduced_tx_set_used : 1;
      uint32_t skip_mode_present : 1;
      uint32_t reserved : 20;
    } bits;
    uint32_t value;
  } mode_control_fields;
  uint8_t cdef_damping_minus_3;
  uint8_t cdef_bits;
  uint8_t cdef_y_strengths[8];
  uint8_t cdef_uv_strengths[8];
  union {
    struct {
      uint16_t yframe_restoration_type : 2;
      uint16_t cbframe_restoration_type : 2;
      uint16_t crframe_restoration_type : 2;
      uint16_t lr_unit_shift : 2;
      uint16_t lr_uv_shift : 1;
      uint16_t reserved : 7;
    } bits;
    uint16_t value;
  } loop_restoration_fields;
  RocdecAv1WarpedMotionParams wm[7];
  uint32_t reserved[8];
} RocdecAv1PicParams;

/*********************************************************************************/
// AV1 slice/tile parameters (VA-API compatible)
/*********************************************************************************/
typedef struct _RocdecAv1SliceParams {
  uint32_t slice_data_size;
  uint32_t slice_data_offset;
  uint32_t slice_data_flag;
  uint16_t tile_row;
  uint16_t tile_column;
  uint16_t tg_start;
  uint16_t tg_end;
  uint8_t anchor_frame_idx;
  uint16_t tile_idx_in_tile_list;
  uint32_t reserved[4];
} RocdecAv1SliceParams;

/*********************************************************************************/
// RocdecPicParams - Picture parameters for decoding
// Used in rocDecDecodeFrame API
/*********************************************************************************/
typedef struct _RocdecPicParams {
  int pic_width;         /**< IN: Coded frame width */
  int pic_height;        /**< IN: Coded frame height */
  int curr_pic_idx;      /**< IN: Output index of the current picture */
  int field_pic_flag;    /**< IN: 0=frame picture, 1=field picture */
  int bottom_field_flag; /**< IN: 0=top field, 1=bottom field */
  int second_field;      /**< IN: Second field of a complementary field pair */
  // Bitstream data
  uint32_t bitstream_data_len;   /**< IN: Number of bytes in bitstream data buffer */
  const uint8_t* bitstream_data; /**< IN: Ptr to bitstream data for this picture */
  uint32_t num_slices;           /**< IN: Number of slices in this picture */

  int ref_pic_flag;      /**< IN: This picture is a reference picture */
  int intra_pic_flag;    /**< IN: This picture is entirely intra coded */
  uint32_t reserved[30]; /**< Reserved for future use */

  // Codec-specific data
  union {
    RocdecMpeg2PicParams mpeg2; /**< Also used for MPEG-1 */
    RocdecAvcPicParams avc;
    RocdecHevcPicParams hevc;
    RocdecVc1PicParams vc1;
    RocdecJPEGPicParams jpeg;
    RocdecVp9PicParams vp9;
    RocdecAv1PicParams av1;
    uint32_t codec_reserved[256];
  } pic_params;

  // Variable size array - one slice param struct per slice
  union {
    RocdecAvcSliceParams* avc;
    RocdecHevcSliceParams* hevc;
    RocdecVp9SliceParams* vp9;
    RocdecAv1SliceParams* av1;
  } slice_params;

  union {
    RocdecAvcIQMatrix avc;
    RocdecHevcIQMatrix hevc;
  } iq_matrix;
} RocdecPicParams;

/*********************************************************************************/
// RocdecProcParams - Picture parameters for postprocessing
// Used in rocDecGetVideoFrame API
/*********************************************************************************/
typedef struct _RocdecProcParams {
  int progressive_frame;      /**< IN: Input is progressive */
  int top_field_first;        /**< IN: Input frame is top field first */
  uint32_t reserved_flags[2]; /**< Reserved for future use (set to zero) */

  // Fields below are used for raw YUV input
  uint64_t raw_input_dptr;    /**< IN: Input HIP device ptr for raw YUV */
  uint32_t raw_input_pitch;   /**< IN: pitch in bytes of raw YUV input */
  uint32_t raw_input_format;  /**< IN: Input YUV format (rocDecVideoCodec_enum) */
  uint64_t raw_output_dptr;   /**< IN: Output HIP device mem ptr for raw YUV */
  uint32_t raw_output_pitch;  /**< IN: pitch in bytes of raw YUV output */
  uint32_t raw_output_format; /**< IN: Output YUV format (rocDecVideoCodec_enum) */
  uint32_t reserved[16];      /**< Reserved for future use (set to zero) */
} RocdecProcParams;

// ---- API functions ----

extern rocDecStatus ROCDECAPI
rocDecCreateDecoder(rocDecDecoderHandle* decoder_handle,
                    RocDecoderCreateInfo* decoder_create_info);

extern rocDecStatus ROCDECAPI
rocDecDestroyDecoder(rocDecDecoderHandle decoder_handle);

extern rocDecStatus ROCDECAPI
rocDecGetDecoderCaps(RocdecDecodeCaps* decode_caps);

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
