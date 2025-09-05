/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#if __has_include("cuda_runtime.h")

#    include "torch/cuda/tt_cuda.h"

#    include "cuda_runtime.h"



#include <torch/cuda/multi_device.h>
#endif
