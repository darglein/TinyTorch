
#include "torch/core/graph.h"

#include "multi_device.h"

#ifdef TT_HAS_CUDA
#    include "ops_impl_cuda_helper.h"
#    include "torch/core/ops/ops_operators.h"
#    include "torch/core/ops/ops_tensor_creation.h"
#    include "torch/cuda/ops_impl_cuda_helper.h"

namespace tinytorch
{
namespace cuda
{
template <int NUM_GPUS, typename T>
struct MultiGPUInputSimple
{
    __host__ MultiGPUInputSimple(const MultiDeviceTensor& data)
    {
        CHECK_GE(data.size(), NUM_GPUS);
        for (int i = 0; i < NUM_GPUS; ++i)
        {
            in_x[i] = data.data[i].data_ptr<T>();

            if (i == 0)
            {
                N = data.data[i].size(0);
            }
            else
            {
                CHECK_EQ(N, data.data[i].size(0));
            }
        }
    }

    __host__ MultiGPUInputSimple(const std::vector<Tensor>& data)
    {
        CHECK_GE(data.size(), NUM_GPUS);
        for (int i = 0; i < NUM_GPUS; ++i)
        {
            in_x[i] = data[i].data_ptr<T>();

            if (i == 0)
            {
                N = data[i].size(0);
            }
            else
            {
                CHECK_EQ(N, data[i].size(0));
            }
        }
    }

    __device__ T CombinedRead(int tid)
    {
        CUDA_KERNEL_ASSERT(tid >= 0 && tid < N);
        T result = 0;
        for (int i = 0; i < NUM_GPUS; ++i)
        {
            result += in_x[i][tid];
        }
        return result;
    }


    __device__ void WriteToAll(T result, int tid)
    {
        CUDA_KERNEL_ASSERT(tid >= 0 && tid < N);
        for (int i = 0; i < NUM_GPUS; ++i)
        {
            in_x[i][tid] = result;
        }
    }

    __device__ void WriteToD0(T result, int tid)
    {
        CUDA_KERNEL_ASSERT(tid >= 0 && tid < N);

        in_x[0][tid] = result;
    }

   public:
    T* in_x[NUM_GPUS];
    int64_t N = 0;
};

template <typename T, int NUM_INPUTS>
static __global__ void ReduceToDevice0(MultiGPUInputSimple<NUM_INPUTS, T> input)
{
    int64_t tid = int64_t(blockIdx.x) * 128 + threadIdx.x;

    if (tid >= input.N)
    {
        return;
    }
    auto sum = input.CombinedRead(tid);
    input.WriteToD0(sum, tid);
}

template <typename T>
static __global__ void Simplecopy(T* src, T* dst, int64_t N)
{
    int64_t tid = int64_t(blockIdx.x) * 128 + threadIdx.x;

    if (tid >= N)
    {
        return;
    }
    dst[tid] = src[tid];
}


void MultiDeviceTensor::ReduceGradientSumToMainUVA()
{
    std::vector<Tensor> grads;
    for (auto& d : data)
    {
        grads.push_back(d.mutable_grad());
    }

    int num_gpus  = size();
    int64_t numel = Main().numel();
    DeviceGuard dg(data[0].device());
    switch (num_gpus)
    {
        case 1:
            break;
        case 2:
            ReduceToDevice0<<<iDivUp(numel, 128), 128, 0, getCurrentCUDAStream()>>>(
                MultiGPUInputSimple<2, float>(grads));
            break;
        case 3:
            ReduceToDevice0<<<iDivUp(numel, 128), 128, 0, getCurrentCUDAStream()>>>(
                MultiGPUInputSimple<3, float>(grads));
            break;
        case 4:
            ReduceToDevice0<<<iDivUp(numel, 128), 128, 0, getCurrentCUDAStream()>>>(
                MultiGPUInputSimple<4, float>(grads));
            break;
        default:
            CHECK(false);
    }
    CUDA_SYNC_CHECK_ERROR();
}

void MultiDeviceTensor::ReduceSumToMainUVA()
{
    int num_gpus  = size();
    int64_t numel = Main().numel();
    DeviceGuard dg(data[0].device());
    switch (num_gpus)
    {
        case 1:
            break;
        case 2:
            ReduceToDevice0<<<iDivUp(numel, 128), 128, 0, getCurrentCUDAStream()>>>(
                MultiGPUInputSimple<2, float>(*this));
            break;
        case 3:
            ReduceToDevice0<<<iDivUp(numel, 128), 128, 0, getCurrentCUDAStream()>>>(
                MultiGPUInputSimple<3, float>(*this));
            break;
        case 4:
            ReduceToDevice0<<<iDivUp(numel, 128), 128, 0, getCurrentCUDAStream()>>>(
                MultiGPUInputSimple<4, float>(*this));
            break;
        default:
            CHECK(false);
    }
    CUDA_SYNC_CHECK_ERROR();
}

void MultiDeviceTensor::copy_parameters_from_main_to_others_uva()
{
    for (int i = 1; i < data.size(); ++i)
    {
        auto src = data[0].data_ptr<float>();
        auto dst = data[i].data_ptr<float>();

        int64_t numel = data[0].numel();

        DeviceGuard dg(data[i].device());

        Simplecopy<<<iDivUp(numel, 128), 128, 0, getCurrentCUDAStream()>>>(src, dst, numel);
        CUDA_SYNC_CHECK_ERROR();
    }
}


}  // namespace cuda
}  // namespace tinytorch

#endif