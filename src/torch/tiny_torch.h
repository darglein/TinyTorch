/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once



#include "torch/core/backward.h"
#include "torch/core/graph.h"
#include "torch/core/ops/all.h"
#include "torch/core/optimizer.h"
#include "torch/core/tensor.h"
#include "torch/core/module.h"

#include "torch/tiny_torch_config.h"

#define TINY_TORCH

#ifdef MAX_TENSORINFO_DIMS
#error asdf
#endif


