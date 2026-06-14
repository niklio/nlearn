// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PJRT_PLUGIN_PJRT_METAL_CLIENT_H_
#define IREE_PJRT_PLUGIN_PJRT_METAL_CLIENT_H_

#include "iree/hal/drivers/metal/api.h"
#include "iree_pjrt/common/api_impl.h"

namespace iree::pjrt::metal {

class MetalClientInstance final : public ClientInstance {
 public:
  MetalClientInstance(std::unique_ptr<Platform> platform);
  ~MetalClientInstance();
  iree_status_t CreateDriver(iree_hal_driver_t** out_driver) override;
  bool SetDefaultCompilerFlags(CompilerJob* compiler_job) override;

 private:
};

}  // namespace iree::pjrt::metal

#endif  // IREE_PJRT_PLUGIN_PJRT_METAL_CLIENT_H_
