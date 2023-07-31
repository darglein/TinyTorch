/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "graph.h"
namespace tinytorch
{
int autograd::Node::current_seq_nr = 0;

static thread_local bool grad_mode_ = true;

bool GradMode::is_enabled()
{
    return grad_mode_;
}
void GradMode::set_enabled(bool enabled)
{
    grad_mode_ = enabled;
}

}  // namespace tinytorch