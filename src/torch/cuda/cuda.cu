/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "cuda.h"


namespace tinytorch
{
namespace cuda
{

static cudaStream_t& thread_local_stream()
{
    static thread_local cudaStream_t strm = 0;
    return strm;
}


cudaStream_t getCurrentCUDAStream()
{
    return thread_local_stream();
}
void setCUDAStreamForThisThread(cudaStream_t stream)
{
    thread_local_stream() = stream;
}

}  // namespace cuda
}  // namespace tinytorch