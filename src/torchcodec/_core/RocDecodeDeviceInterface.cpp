// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Use c10/hip headers instead of c10/cuda to avoid <cuda.h> dependency
// when compiling with GCC on ROCm. On ROCm PyTorch, the HIP stream
// API is the native interface.
#include <c10/hip/HIPStream.h>
#include <torch/types.h>
#include <map>
#include <mutex>
#include <vector>

#include "RocDecodeDeviceInterface.h"

#include "DeviceInterface.h"
#include "FFMPEGCommon.h"
#include "RocDecodeCache.h"
#include "RocDecodeRuntimeLoader.h"

#include "rocdecode_include/rocdecode.h"
#include "rocdecode_include/rocparser.h"

extern "C" {
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {

namespace {

// Register as the "beta" variant for CUDA device type.
// On ROCm PyTorch, torch::kCUDA maps to HIP devices. The build system
// ensures only one of NVDEC or rocDecode is compiled in, so there's
// no registration conflict.
static bool g_rocm_beta = registerDeviceInterface(
    DeviceInterfaceKey(torch::kCUDA, /*variant=*/"beta"),
    [](const torch::Device& device) {
      return new RocDecodeDeviceInterface(device);
    });

// Also register as the default "ffmpeg" CUDA variant so that
// device="cuda" without explicit variant selection works on ROCm.
static bool g_rocm_default = registerDeviceInterface(
    DeviceInterfaceKey(torch::kCUDA),
    [](const torch::Device& device) {
      return new RocDecodeDeviceInterface(device);
    });

// ---- Parser callbacks (C-style, forwarded to instance methods) ----
static int ROCDECAPI
pfnSequenceCallback(void* pUserData, RocdecVideoFormat* videoFormat) {
  auto decoder = static_cast<RocDecodeDeviceInterface*>(pUserData);
  return decoder->streamPropertyChange(videoFormat);
}

static int ROCDECAPI
pfnDecodePictureCallback(void* pUserData, RocdecPicParams* picParams) {
  auto decoder = static_cast<RocDecodeDeviceInterface*>(pUserData);
  return decoder->frameReadyForDecoding(picParams);
}

static int ROCDECAPI
pfnDisplayPictureCallback(void* pUserData, RocdecParserDispInfo* dispInfo) {
  auto decoder = static_cast<RocDecodeDeviceInterface*>(pUserData);
  return decoder->frameReadyInDisplayOrder(dispInfo);
}

// ---- Codec validation ----
std::optional<rocDecVideoCodec> validateCodecSupport(AVCodecID codecId) {
  switch (codecId) {
    case AV_CODEC_ID_H264:
      return rocDecVideoCodec_AVC;
    case AV_CODEC_ID_HEVC:
      return rocDecVideoCodec_HEVC;
    case AV_CODEC_ID_AV1:
      return rocDecVideoCodec_AV1;
    case AV_CODEC_ID_VP9:
      return rocDecVideoCodec_VP9;
    case AV_CODEC_ID_VP8:
      return rocDecVideoCodec_VP8;
    case AV_CODEC_ID_MPEG4:
      return rocDecVideoCodec_MPEG4;
    case AV_CODEC_ID_MPEG1VIDEO:
      return rocDecVideoCodec_MPEG1;
    case AV_CODEC_ID_MPEG2VIDEO:
      return rocDecVideoCodec_MPEG2;
    default:
      return std::nullopt;
  }
}

std::optional<rocDecVideoChromaFormat> validateChromaSupport(
    const AVPixFmtDescriptor* desc) {
  TORCH_CHECK(desc != nullptr, "desc can't be null");

  if (desc->nb_components == 1) {
    return rocDecVideoChromaFormat_Monochrome;
  } else if (
      desc->nb_components >= 3 && !(desc->flags & AV_PIX_FMT_FLAG_RGB)) {
    if (desc->log2_chroma_w == 0 && desc->log2_chroma_h == 0) {
      return rocDecVideoChromaFormat_444;
    } else if (desc->log2_chroma_w == 1 && desc->log2_chroma_h == 1) {
      return rocDecVideoChromaFormat_420;
    } else if (desc->log2_chroma_w == 1 && desc->log2_chroma_h == 0) {
      return rocDecVideoChromaFormat_422;
    }
  }

  return std::nullopt;
}

bool nativeRocDecSupport(
    const torch::Device& device,
    const SharedAVCodecContext& codecContext) {
  auto codecType = validateCodecSupport(codecContext->codec_id);
  if (!codecType.has_value()) {
    return false;
  }

  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(codecContext->pix_fmt);
  if (!desc) {
    return false;
  }

  auto chromaFormat = validateChromaSupport(desc);
  if (!chromaFormat.has_value()) {
    return false;
  }

  // Ensure rocDecode queries the correct GPU
  int deviceIndex = getDeviceIndex_HIP(device);
  hipSetDevice(deviceIndex);

  // Query decoder capabilities
  RocdecDecodeCaps caps = {};
  caps.device_id = static_cast<uint8_t>(deviceIndex);
  caps.codec_type = codecType.value();
  caps.chroma_format = chromaFormat.value();
  caps.bit_depth_minus_8 = static_cast<uint32_t>(desc->comp[0].depth - 8);

  rocDecStatus result = rocDecGetDecoderCaps(&caps);
  if (result != ROCDEC_SUCCESS) {
    return false;
  }

  if (!caps.is_supported) {
    return false;
  }

  auto coded_width = static_cast<uint32_t>(codecContext->coded_width);
  auto coded_height = static_cast<uint32_t>(codecContext->coded_height);
  if (coded_width < caps.min_width || coded_height < caps.min_height ||
      coded_width > caps.max_width || coded_height > caps.max_height) {
    return false;
  }

  // Check NV12 output format support
  bool supportsNV12 =
      (caps.output_format_mask >> rocDecVideoSurfaceFormat_NV12) & 1;
  if (!supportsNV12) {
    return false;
  }

  return true;
}

// Callback for freeing HIP memory associated with AVFrame
void hipBufferFreeCallback(void* opaque, [[maybe_unused]] uint8_t* data) {
  hipFree(opaque);
}

} // namespace

// ---- Constructor / Destructor ----

RocDecodeDeviceInterface::RocDecodeDeviceInterface(const torch::Device& device)
    : DeviceInterface(device) {
  TORCH_CHECK(
      g_rocm_beta || g_rocm_default,
      "RocDecodeDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == torch::kCUDA, "Unsupported device: ", device_.str());

  initializeHIPContextWithPytorch(device_);

  // Explicitly set the HIP device so the rocDecode library (loaded via
  // dlopen) uses the same GPU as PyTorch. Without this, rocDecode may
  // open a DRM fd to a different GPU, causing memory access faults in
  // multi-GPU environments.
  int deviceIndex = getDeviceIndex_HIP(device_);
  hipError_t hipErr = hipSetDevice(deviceIndex);
  TORCH_CHECK(
      hipErr == hipSuccess,
      "Failed to set HIP device ",
      deviceIndex,
      ": ",
      hipGetErrorString(hipErr));

  rocDecodeAvailable_ = loadRocDecodeLibrary();
}

RocDecodeDeviceInterface::~RocDecodeDeviceInterface() {
  if (decoder_) {
    flush();
    RocDecodeCache::getCache(device_).returnDecoder(
        &videoFormat_, std::move(decoder_));
  }

  if (videoParser_) {
    rocDecDestroyVideoParser(videoParser_);
    videoParser_ = nullptr;
  }
}

// ---- Initialization ----

void RocDecodeDeviceInterface::initialize(
    const AVStream* avStream,
    const UniqueDecodingAVFormatContext& avFormatCtx,
    [[maybe_unused]] const SharedAVCodecContext& codecContext) {
  if (!rocDecodeAvailable_ || !nativeRocDecSupport(device_, codecContext)) {
    cpuFallback_ = createDeviceInterface(torch::kCPU);
    TORCH_CHECK(
        cpuFallback_ != nullptr, "Failed to create CPU device interface");
    cpuFallback_->initialize(avStream, avFormatCtx, codecContext);
    cpuFallback_->initializeVideo(
        VideoStreamOptions(),
        {},
        /*resizedOutputDims=*/std::nullopt);
    return;
  }

  TORCH_CHECK(avStream != nullptr, "AVStream cannot be null");
  timeBase_ = avStream->time_base;
  frameRateAvgFromFFmpeg_ = avStream->r_frame_rate;

  const AVCodecParameters* codecPar = avStream->codecpar;
  TORCH_CHECK(codecPar != nullptr, "CodecParameters cannot be null");

  initializeBSF(codecPar, avFormatCtx);

  // Create parser
  RocdecParserParams parserParams = {};
  auto codecType = validateCodecSupport(codecPar->codec_id);
  TORCH_CHECK(
      codecType.has_value(),
      "This should never happen, we should be using the CPU fallback. "
      "Please report a bug.");
  parserParams.codec_type = codecType.value();
  parserParams.max_num_decode_surfaces = 8;
  parserParams.max_display_delay = 0;
  parserParams.clock_rate = 0; // default 10MHz
  parserParams.user_data = this;
  parserParams.pfn_sequence_callback = pfnSequenceCallback;
  parserParams.pfn_decode_picture = pfnDecodePictureCallback;
  parserParams.pfn_display_picture = pfnDisplayPictureCallback;

  rocDecStatus result =
      rocDecCreateVideoParser(&videoParser_, &parserParams);
  TORCH_CHECK(
      result == ROCDEC_SUCCESS,
      "Failed to create rocDecode video parser: ",
      rocDecGetErrorName(result));
}

void RocDecodeDeviceInterface::initializeBSF(
    const AVCodecParameters* codecPar,
    const UniqueDecodingAVFormatContext& avFormatCtx) {
  // Setup bitstream filters - identical logic to BetaCudaDeviceInterface
  TORCH_CHECK(codecPar != nullptr, "codecPar cannot be null");
  TORCH_CHECK(avFormatCtx != nullptr, "AVFormatContext cannot be null");
  TORCH_CHECK(
      avFormatCtx->iformat != nullptr,
      "AVFormatContext->iformat cannot be null");
  std::string filterName;

  switch (codecPar->codec_id) {
    case AV_CODEC_ID_H264: {
      const std::string formatName = avFormatCtx->iformat->long_name
          ? avFormatCtx->iformat->long_name
          : "";
      if (formatName == "QuickTime / MOV" ||
          formatName == "FLV (Flash Video)" ||
          formatName == "Matroska / WebM" || formatName == "raw H.264 video") {
        filterName = "h264_mp4toannexb";
      }
      break;
    }
    case AV_CODEC_ID_HEVC: {
      const std::string formatName = avFormatCtx->iformat->long_name
          ? avFormatCtx->iformat->long_name
          : "";
      if (formatName == "QuickTime / MOV" ||
          formatName == "FLV (Flash Video)" ||
          formatName == "Matroska / WebM" || formatName == "raw HEVC video") {
        filterName = "hevc_mp4toannexb";
      }
      break;
    }
    case AV_CODEC_ID_MPEG4: {
      const std::string formatName =
          avFormatCtx->iformat->name ? avFormatCtx->iformat->name : "";
      if (formatName == "avi") {
        filterName = "mpeg4_unpack_bframes";
      }
      break;
    }
    default:
      break;
  }

  if (filterName.empty()) {
    return;
  }

  const AVBitStreamFilter* avBSF = av_bsf_get_by_name(filterName.c_str());
  TORCH_CHECK(
      avBSF != nullptr, "Failed to find bitstream filter: ", filterName);

  AVBSFContext* avBSFContext = nullptr;
  int retVal = av_bsf_alloc(avBSF, &avBSFContext);
  TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to allocate bitstream filter: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  bitstreamFilter_.reset(avBSFContext);

  retVal = avcodec_parameters_copy(bitstreamFilter_->par_in, codecPar);
  TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to copy codec parameters: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  retVal = av_bsf_init(bitstreamFilter_.get());
  TORCH_CHECK(
      retVal == AVSUCCESS,
      "Failed to initialize bitstream filter: ",
      getFFMPEGErrorStringFromErrorCode(retVal));
}

// ---- Decoder creation ----

UniqueRocDecoder RocDecodeDeviceInterface::createDecoder(
    RocdecVideoFormat* videoFormat,
    int deviceId) {
  // Set HIP device before creating the decoder to ensure rocDecode
  // uses the same GPU context as PyTorch/HIP.
  hipSetDevice(deviceId);

  RocDecoderCreateInfo decoderParams = {};
  // device_id is the FIRST field in the real v1.5.0 struct (uint8_t)
  decoderParams.device_id = static_cast<uint8_t>(deviceId);
  decoderParams.width = videoFormat->coded_width;
  decoderParams.height = videoFormat->coded_height;
  decoderParams.num_decode_surfaces = videoFormat->min_num_decode_surfaces;
  decoderParams.codec_type = videoFormat->codec;
  decoderParams.chroma_format = videoFormat->chroma_format;
  decoderParams.bit_depth_minus_8 = videoFormat->bit_depth_luma_minus8;
  decoderParams.intra_decode_only = 0;
  decoderParams.max_width = videoFormat->coded_width;
  decoderParams.max_height = videoFormat->coded_height;
  decoderParams.display_rect.left =
      static_cast<int16_t>(videoFormat->display_area.left);
  decoderParams.display_rect.top =
      static_cast<int16_t>(videoFormat->display_area.top);
  decoderParams.display_rect.right =
      static_cast<int16_t>(videoFormat->display_area.right);
  decoderParams.display_rect.bottom =
      static_cast<int16_t>(videoFormat->display_area.bottom);
  // Request NV12 output - same as NVDEC path. 10bit videos will be
  // automatically converted to 8bit by the VCN hardware.
  decoderParams.output_format = rocDecVideoSurfaceFormat_NV12;
  decoderParams.target_width = static_cast<uint32_t>(
      videoFormat->display_area.right - videoFormat->display_area.left);
  decoderParams.target_height = static_cast<uint32_t>(
      videoFormat->display_area.bottom - videoFormat->display_area.top);
  decoderParams.num_output_surfaces = 1;

  rocDecDecoderHandle* decoderHandle = new rocDecDecoderHandle();
  rocDecStatus result = rocDecCreateDecoder(decoderHandle, &decoderParams);
  TORCH_CHECK(
      result == ROCDEC_SUCCESS,
      "Failed to create rocDecode decoder: ",
      rocDecGetErrorName(result));

  return UniqueRocDecoder(decoderHandle, RocDecoderDeleter{});
}

// ---- Parser callbacks ----

int RocDecodeDeviceInterface::streamPropertyChange(
    RocdecVideoFormat* videoFormat) {
  TORCH_CHECK(videoFormat != nullptr, "Invalid video format");

  videoFormat_ = *videoFormat;

  if (videoFormat_.min_num_decode_surfaces == 0) {
    videoFormat_.min_num_decode_surfaces = 20;
  }

  if (!decoder_) {
    decoder_ = RocDecodeCache::getCache(device_).getDecoder(videoFormat);

    if (!decoder_) {
      decoder_ = createDecoder(
          videoFormat, getDeviceIndex_HIP(device_));
    }

    TORCH_CHECK(decoder_, "Failed to get or create rocDecode decoder");
  }

  return static_cast<int>(videoFormat_.min_num_decode_surfaces);
}

int RocDecodeDeviceInterface::frameReadyForDecoding(
    RocdecPicParams* picParams) {
  TORCH_CHECK(picParams != nullptr, "Invalid picture parameters");
  TORCH_CHECK(decoder_, "Decoder not initialized before picture decode");

  rocDecStatus result = rocDecDecodeFrame(*decoder_.get(), picParams);

  // 0 means error, 1 means success (same convention as NVCUVID)
  return (result == ROCDEC_SUCCESS) ? 1 : 0;
}

int RocDecodeDeviceInterface::frameReadyInDisplayOrder(
    RocdecParserDispInfo* dispInfo) {
  readyFrames_.push(*dispInfo);
  return 1; // success
}

// ---- Send/Receive pattern ----

int RocDecodeDeviceInterface::sendPacket(ReferenceAVPacket& packet) {
  if (cpuFallback_) {
    return cpuFallback_->sendPacket(packet);
  }

  TORCH_CHECK(
      packet.get() && packet->data && packet->size > 0,
      "sendPacket received an empty packet, this is unexpected.");

  // Apply BSF if needed
  AutoAVPacket filteredAutoPacket;
  ReferenceAVPacket filteredPacket(filteredAutoPacket);
  ReferenceAVPacket& packetToSend = applyBSF(packet, filteredPacket);

  RocdecSourceDataPacket rocPacket = {};
  rocPacket.payload = packetToSend->data;
  rocPacket.payload_size = static_cast<uint32_t>(packetToSend->size);
  rocPacket.flags = ROCDEC_PKT_TIMESTAMP;
  rocPacket.pts = static_cast<uint64_t>(packetToSend->pts);

  return sendRocDecPacket(rocPacket);
}

int RocDecodeDeviceInterface::sendEOFPacket() {
  if (cpuFallback_) {
    return cpuFallback_->sendEOFPacket();
  }

  RocdecSourceDataPacket rocPacket = {};
  rocPacket.flags = ROCDEC_PKT_ENDOFSTREAM;
  eofSent_ = true;

  return sendRocDecPacket(rocPacket);
}

int RocDecodeDeviceInterface::sendRocDecPacket(
    RocdecSourceDataPacket& rocPacket) {
  rocDecStatus result = rocDecParseVideoData(videoParser_, &rocPacket);
  return result == ROCDEC_SUCCESS ? AVSUCCESS : AVERROR_EXTERNAL;
}

ReferenceAVPacket& RocDecodeDeviceInterface::applyBSF(
    ReferenceAVPacket& packet,
    ReferenceAVPacket& filteredPacket) {
  if (!bitstreamFilter_) {
    return packet;
  }

  int retVal = av_bsf_send_packet(bitstreamFilter_.get(), packet.get());
  TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to send packet to bitstream filter: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  retVal = av_bsf_receive_packet(bitstreamFilter_.get(), filteredPacket.get());
  TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to receive packet from bitstream filter: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  return filteredPacket;
}

// ---- Frame retrieval ----

int RocDecodeDeviceInterface::receiveFrame(UniqueAVFrame& avFrame) {
  if (cpuFallback_) {
    return cpuFallback_->receiveFrame(avFrame);
  }

  if (readyFrames_.empty()) {
    return eofSent_ ? AVERROR_EOF : AVERROR(EAGAIN);
  }

  RocdecParserDispInfo dispInfo = readyFrames_.front();
  readyFrames_.pop();

  // Ensure the correct HIP device is active before calling rocDecode APIs
  hipSetDevice(getDeviceIndex_HIP(device_));

  // rocDecGetVideoFrame is a blocking call that returns per-plane device
  // pointers. Unlike NVDEC's map/unmap pattern, rocDecode gives us direct
  // pointers to the decoded frame data.
  RocdecProcParams procParams = {};
  procParams.progressive_frame = dispInfo.progressive_frame;
  procParams.top_field_first = dispInfo.top_field_first;

  void* devMemPtr[3] = {nullptr, nullptr, nullptr};
  uint32_t pitch = 0;

  rocDecStatus result = rocDecGetVideoFrame(
      *decoder_.get(), dispInfo.picture_index, devMemPtr, &pitch, &procParams);
  if (result != ROCDEC_SUCCESS) {
    return AVERROR_EXTERNAL;
  }

  avFrame = convertRocDecFrameToAVFrame(devMemPtr, pitch, dispInfo);

  return AVSUCCESS;
}

UniqueAVFrame RocDecodeDeviceInterface::convertRocDecFrameToAVFrame(
    void* devMemPtr[3],
    uint32_t pitch,
    const RocdecParserDispInfo& dispInfo) {
  TORCH_CHECK(devMemPtr[0] != nullptr, "Invalid decoded frame pointer");

  int width = videoFormat_.display_area.right - videoFormat_.display_area.left;
  int height =
      videoFormat_.display_area.bottom - videoFormat_.display_area.top;

  TORCH_CHECK(width > 0 && height > 0, "Invalid frame dimensions");
  TORCH_CHECK(
      pitch >= static_cast<uint32_t>(width), "Pitch must be >= width");

  // Unlike NVDEC which has an explicit map/unmap pattern to lock decode
  // surfaces, rocDecGetVideoFrame returns raw pointers into the decoder's
  // internal surface pool. These surfaces can be recycled as soon as the
  // next frame is decoded, so we must copy the NV12 data to owned memory
  // before returning.
  int yHeight = height;
  int uvHeight = height / 2; // NV12: UV plane is half height
  size_t ySize = static_cast<size_t>(pitch) * yHeight;
  size_t uvSize = static_cast<size_t>(pitch) * uvHeight;
  size_t totalSize = ySize + uvSize;

  uint8_t* ownedBuffer = nullptr;
  hipError_t err =
      hipMalloc(reinterpret_cast<void**>(&ownedBuffer), totalSize);
  TORCH_CHECK(
      err == hipSuccess,
      "Failed to allocate HIP memory for frame copy: ",
      hipGetErrorString(err));

  hipStream_t currentStream =
      c10::hip::getCurrentHIPStream(device_.index()).stream();

  // Copy Y plane from decoder surface to owned buffer
  err = hipMemcpy2DAsync(
      ownedBuffer,
      pitch,
      devMemPtr[0],
      pitch,
      width,
      yHeight,
      hipMemcpyDeviceToDevice,
      currentStream);
  TORCH_CHECK(
      err == hipSuccess,
      "Failed to copy Y plane: ",
      hipGetErrorString(err));

  // Copy UV plane from decoder surface to owned buffer
  err = hipMemcpy2DAsync(
      ownedBuffer + ySize,
      pitch,
      devMemPtr[1],
      pitch,
      width,
      uvHeight,
      hipMemcpyDeviceToDevice,
      currentStream);
  TORCH_CHECK(
      err == hipSuccess,
      "Failed to copy UV plane: ",
      hipGetErrorString(err));

  UniqueAVFrame avFrame(av_frame_alloc());
  TORCH_CHECK(avFrame.get() != nullptr, "Failed to allocate AVFrame");

  avFrame->width = width;
  avFrame->height = height;
  avFrame->format = AV_PIX_FMT_CUDA; // CUDA format on ROCm = HIP device mem
  avFrame->pts = static_cast<int64_t>(dispInfo.pts);

  setDuration(avFrame, computeSafeDuration(frameRateAvgFromFFmpeg_, timeBase_));

  // Map matrix_coefficients to AVColorSpace
  switch (videoFormat_.video_signal_description.matrix_coefficients) {
    case 1:
      avFrame->colorspace = AVCOL_SPC_BT709;
      break;
    case 6:
      avFrame->colorspace = AVCOL_SPC_SMPTE170M; // BT.601
      break;
    default:
      avFrame->colorspace = AVCOL_SPC_SMPTE170M;
      break;
  }

  avFrame->color_range =
      videoFormat_.video_signal_description.video_full_range_flag
      ? AVCOL_RANGE_JPEG
      : AVCOL_RANGE_MPEG;

  // Point AVFrame at our owned copy of the NV12 data
  avFrame->data[0] = ownedBuffer;
  avFrame->data[1] = ownedBuffer + ySize;
  avFrame->data[2] = nullptr;
  avFrame->data[3] = nullptr;
  avFrame->linesize[0] = static_cast<int>(pitch);
  avFrame->linesize[1] = static_cast<int>(pitch);
  avFrame->linesize[2] = 0;
  avFrame->linesize[3] = 0;

  // Register cleanup callback so HIP memory is freed when AVFrame is released
  avFrame->opaque_ref = av_buffer_create(
      nullptr,
      0,
      hipBufferFreeCallback,
      ownedBuffer,
      0);
  TORCH_CHECK(
      avFrame->opaque_ref != nullptr,
      "Failed to create GPU memory cleanup reference");

  return avFrame;
}

// ---- Flush ----

void RocDecodeDeviceInterface::flush() {
  if (cpuFallback_) {
    cpuFallback_->flush();
    return;
  }

  // Send EOF to flush remaining frames from parser
  sendEOFPacket();
  eofSent_ = false;

  // Clear the ready frames queue
  std::queue<RocdecParserDispInfo> emptyQueue;
  std::swap(readyFrames_, emptyQueue);
}

// ---- CPU fallback: transfer CPU frame to GPU as NV12 ----

UniqueAVFrame RocDecodeDeviceInterface::transferCpuFrameToGpuNV12(
    UniqueAVFrame& cpuFrame) {
  TORCH_CHECK(cpuFrame != nullptr, "CPU frame cannot be null");

  int width = cpuFrame->width;
  int height = cpuFrame->height;

  // Intermediate NV12 CPU frame
  UniqueAVFrame nv12CpuFrame(av_frame_alloc());
  TORCH_CHECK(nv12CpuFrame != nullptr, "Failed to allocate NV12 CPU frame");

  nv12CpuFrame->format = AV_PIX_FMT_NV12;
  nv12CpuFrame->width = width;
  nv12CpuFrame->height = height;

  int ret = av_frame_get_buffer(nv12CpuFrame.get(), 0);
  TORCH_CHECK(
      ret >= 0,
      "Failed to allocate NV12 CPU frame buffer: ",
      getFFMPEGErrorStringFromErrorCode(ret));

  SwsFrameContext swsFrameContext(
      width,
      height,
      static_cast<AVPixelFormat>(cpuFrame->format),
      width,
      height);

  if (!swsContext_ || prevSwsFrameContext_ != swsFrameContext) {
    swsContext_ = createSwsContext(
        swsFrameContext, cpuFrame->colorspace, AV_PIX_FMT_NV12, SWS_BILINEAR);
    prevSwsFrameContext_ = swsFrameContext;
  }

  int convertedHeight = sws_scale(
      swsContext_.get(),
      cpuFrame->data,
      cpuFrame->linesize,
      0,
      height,
      nv12CpuFrame->data,
      nv12CpuFrame->linesize);
  TORCH_CHECK(
      convertedHeight == height, "sws_scale failed for CPU->NV12 conversion");

  int ySize = width * height;
  TORCH_CHECK(ySize % 2 == 0, "Y plane size must be even.");
  int uvSize = ySize / 2;
  size_t totalSize = static_cast<size_t>(ySize + uvSize);

  uint8_t* hipBuffer = nullptr;
  hipError_t err =
      hipMalloc(reinterpret_cast<void**>(&hipBuffer), totalSize);
  TORCH_CHECK(
      err == hipSuccess,
      "Failed to allocate HIP memory: ",
      hipGetErrorString(err));

  UniqueAVFrame gpuFrame(av_frame_alloc());
  TORCH_CHECK(gpuFrame != nullptr, "Failed to allocate GPU AVFrame");

  gpuFrame->format = AV_PIX_FMT_CUDA;
  gpuFrame->width = width;
  gpuFrame->height = height;
  gpuFrame->data[0] = hipBuffer;
  gpuFrame->data[1] = hipBuffer + ySize;
  gpuFrame->linesize[0] = width;
  gpuFrame->linesize[1] = width;

  // Copy Y plane
  err = hipMemcpy2D(
      gpuFrame->data[0],
      gpuFrame->linesize[0],
      nv12CpuFrame->data[0],
      nv12CpuFrame->linesize[0],
      width,
      height,
      hipMemcpyHostToDevice);
  TORCH_CHECK(
      err == hipSuccess,
      "Failed to copy Y plane to GPU: ",
      hipGetErrorString(err));

  // Copy UV plane
  TORCH_CHECK(height % 2 == 0, "Height must be even.");
  err = hipMemcpy2D(
      gpuFrame->data[1],
      gpuFrame->linesize[1],
      nv12CpuFrame->data[1],
      nv12CpuFrame->linesize[1],
      width,
      height / 2,
      hipMemcpyHostToDevice);
  TORCH_CHECK(
      err == hipSuccess,
      "Failed to copy UV plane to GPU: ",
      hipGetErrorString(err));

  ret = av_frame_copy_props(gpuFrame.get(), cpuFrame.get());
  TORCH_CHECK(
      ret >= 0,
      "Failed to copy frame properties: ",
      getFFMPEGErrorStringFromErrorCode(ret));

  // Associate cleanup callback to free HIP memory when AVFrame is freed
  gpuFrame->opaque_ref = av_buffer_create(
      nullptr,
      0,
      hipBufferFreeCallback,
      hipBuffer,
      0);
  TORCH_CHECK(
      gpuFrame->opaque_ref != nullptr,
      "Failed to create GPU memory cleanup reference");

  return gpuFrame;
}

// ---- Color conversion: NV12 -> RGB tensor ----

void RocDecodeDeviceInterface::convertAVFrameToFrameOutput(
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  UniqueAVFrame gpuFrame =
      cpuFallback_ ? transferCpuFrameToGpuNV12(avFrame) : std::move(avFrame);

  TORCH_CHECK(
      gpuFrame->format == AV_PIX_FMT_CUDA,
      "Expected CUDA/HIP format frame from rocDecode interface");

  validatePreAllocatedTensorShape_HIP(
      preAllocatedOutputTensor, gpuFrame->width, gpuFrame->height);

  // Get current HIP stream for synchronization
  hipStream_t currentStream =
      c10::hip::getCurrentHIPStream(device_.index()).stream();

  frameOutput.data = convertNV12FrameToRGB_HIP(
      gpuFrame->data[0],
      gpuFrame->linesize[0],
      gpuFrame->width,
      gpuFrame->height,
      gpuFrame->colorspace,
      gpuFrame->color_range,
      device_,
      currentStream,
      preAllocatedOutputTensor);
}

// ---- Details ----

std::string RocDecodeDeviceInterface::getDetails() {
  std::string details = "rocDecode Device Interface.";
  if (cpuFallback_) {
    details += " Using CPU fallback.";
    if (!rocDecodeAvailable_) {
      details += " rocDecode not available!";
    }
  } else {
    details += " Using AMD VCN hardware decoder.";
  }
  return details;
}

} // namespace facebook::torchcodec
