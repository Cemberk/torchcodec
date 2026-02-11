// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

namespace facebook::torchcodec {

// Dynamically loads librocdecode.so and librocdec_parser.so at runtime.
// Returns true if all required functions were successfully loaded.
// See the corresponding .cpp for the full design rationale, which mirrors
// the NVCUVID runtime loader pattern.
bool loadRocDecodeLibrary();

} // namespace facebook::torchcodec
