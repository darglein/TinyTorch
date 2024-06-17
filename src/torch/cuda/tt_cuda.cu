/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "tt_cuda.h"
#include "ops_impl_cuda_helper.h"

namespace tinytorch
{
namespace cuda
{

static cudaStream_t& thread_local_stream()
{
	static thread_local cudaStream_t strms[16] = {};
    return strms[getDevice()];
}

cudaStream_t getCurrentCUDAStream()
{
    return thread_local_stream();
}

void setCUDAStreamForThisThread(cudaStream_t stream)
{
    thread_local_stream() = stream;
}

int getDevice()
{
	int device_index;
	CHECK_CUDA_ERROR(cudaGetDevice(&device_index));
	return device_index;
}

void setDevice(int device_index)
{
	CHECK_CUDA_ERROR(cudaSetDevice(device_index));
}


}  // namespace cuda
}  // namespace tinytorch