// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// ROCm device interface for TorchCodec using AMD's rocDecode library.
// This is the AMD equivalent of BetaCudaDeviceInterface - it uses
// rocDecode's hardware video decoder (VCN) instead of NVIDIA's NVDEC.
//
// The design closely mirrors BetaCudaDeviceInterface to minimize divergence:
// - Same parser callback pattern (sequence/decode/display)
// - Same send/receive packet architecture
// - Same CPU fallback mechanism
// - Same frame ordering via display callback queue
//
// Key differences from the NVIDIA path:
// - Uses rocDecode API instead of NVCUVID
// - Uses HIP kernels for color conversion instead of NPP
// - rocDecGetVideoFrame returns per-plane device pointers (not CUdeviceptr)
// - rocDecGetVideoFrame is a blocking call (vs cuvidMapVideoFrame)
// - No map/unmap pattern - frames are copied or used directly

#pragma once

#include "DeviceInterface.h"
#include "FFMPEGCommon.h"
#include "HIPCommon.h"
#include "RocDecodeCache.h"

#include "rocdecode_include/rocdecode.h"
#include "rocdecode_include/rocparser.h"

#include <memory>
#include <mutex>
#include <queue>

namespace facebook::torchcodec {

class RocDecodeDeviceInterface : public DeviceInterface {
 public:
  explicit RocDecodeDeviceInterface(const torch::Device& device);
  virtual ~RocDecodeDeviceInterface();

  void initialize(
      const AVStream* avStream,
      const UniqueDecodingAVFormatContext& avFormatCtx,
      const SharedAVCodecContext& codecContext) override;

  void convertAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor) override;

  int sendPacket(ReferenceAVPacket& packet) override;
  int sendEOFPacket() override;
  int receiveFrame(UniqueAVFrame& avFrame) override;
  void flush() override;

  // rocDecode callback functions (must be public for C callbacks)
  int streamPropertyChange(RocdecVideoFormat* videoFormat);
  int frameReadyForDecoding(RocdecPicParams* picParams);
  int frameReadyInDisplayOrder(RocdecParserDispInfo* dispInfo);

  std::string getDetails() override;

 private:
  int sendRocDecPacket(RocdecSourceDataPacket& packet);

  void initializeBSF(
      const AVCodecParameters* codecPar,
      const UniqueDecodingAVFormatContext& avFormatCtx);

  ReferenceAVPacket& applyBSF(
      ReferenceAVPacket& packet,
      ReferenceAVPacket& filteredPacket);

  UniqueAVFrame convertRocDecFrameToAVFrame(
      void* devMemPtr[3],
      uint32_t pitch,
      const RocdecParserDispInfo& dispInfo);

  UniqueAVFrame transferCpuFrameToGpuNV12(UniqueAVFrame& cpuFrame);

  static UniqueRocDecoder createDecoder(
      RocdecVideoFormat* videoFormat,
      int deviceId);

  RocdecVideoParser videoParser_ = nullptr;
  UniqueRocDecoder decoder_;
  RocdecVideoFormat videoFormat_ = {};

  std::queue<RocdecParserDispInfo> readyFrames_;

  bool eofSent_ = false;

  AVRational timeBase_ = {0, 1};
  AVRational frameRateAvgFromFFmpeg_ = {0, 1};

  UniqueAVBSFContext bitstreamFilter_;

  std::unique_ptr<DeviceInterface> cpuFallback_;
  bool rocDecodeAvailable_ = false;
  UniqueSwsContext swsContext_;
  SwsFrameContext prevSwsFrameContext_;
};

} // namespace facebook::torchcodec
