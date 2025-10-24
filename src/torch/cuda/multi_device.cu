
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
template <int NUM_GPUS, typename T, typename ComputeType>
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

    __device__ ComputeType CombinedRead(int tid)
    {
        CUDA_KERNEL_ASSERT(tid >= 0 && tid < N);
        ComputeType result = 0;
        for (int i = 0; i < NUM_GPUS; ++i)
        {
            result += ComputeType(in_x[i][tid]);
        }
        return result;
    }


    __device__ void WriteToAll(ComputeType result, int tid)
    {
        CUDA_KERNEL_ASSERT(tid >= 0 && tid < N);
        for (int i = 0; i < NUM_GPUS; ++i)
        {
            in_x[i][tid] = T(result);
        }
    }

    __device__ void WriteToD0(ComputeType result, int tid)
    {
        CUDA_KERNEL_ASSERT(tid >= 0 && tid < N);

        in_x[0][tid] = T(result);
    }

   public:
    T* in_x[NUM_GPUS];
    int64_t N = 0;
};

template <typename T, typename ComputeType, int NUM_INPUTS>
static __global__ void ReduceToDevice0(MultiGPUInputSimple<NUM_INPUTS, T, ComputeType> input)
{
    int64_t tid = int64_t(blockIdx.x) * 128 + threadIdx.x;

    if (tid >= input.N)
    {
        return;
    }
    auto sum = input.CombinedRead(tid);
    input.WriteToD0(sum, tid);
}

template <typename T, typename ComputeType>
static void ReduceToDevice0Launcher(std::vector<Tensor> grads)
{
    int num_gpus  = grads.size();
    int64_t numel = grads[0].numel();
    DeviceGuard dg(grads[0].device());
    switch (num_gpus)
    {
        case 1:
            break;
        case 2:
            ReduceToDevice0<<<iDivUp(numel, 128), 128, 0, getCurrentCUDAStream()>>>(
                MultiGPUInputSimple<2, T, ComputeType>(grads));
            break;
        case 3:
            ReduceToDevice0<<<iDivUp(numel, 128), 128, 0, getCurrentCUDAStream()>>>(
                MultiGPUInputSimple<3, T, ComputeType>(grads));
            break;
        case 4:
            ReduceToDevice0<<<iDivUp(numel, 128), 128, 0, getCurrentCUDAStream()>>>(
                MultiGPUInputSimple<4, T, ComputeType>(grads));
            break;
        default:
            CHECK(false);
    }
    CUDA_SYNC_CHECK_ERROR();
}

void MultiDeviceTensor::ReduceGradientSumToMainUVA()
{
    std::vector<Tensor> grads;
    for (auto& d : data)
    {
        grads.push_back(d.mutable_grad());
    }

    if (data[0].scalar_type() == kFloat32)
    {
        ReduceToDevice0Launcher<float, float>(grads);
    }
    else if (data[0].scalar_type() == kFloat64)
    {
        ReduceToDevice0Launcher<double, double>(grads);
    }else
    {
        CHECK(false);
    }

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


void MultiDeviceTensor::ReduceSumToMainUVA()
{
    if (data[0].scalar_type() == kFloat16)
    {
        ReduceToDevice0Launcher<__half, float>(this->data);
    }
    else if (data[0].scalar_type() == kFloat32)
    {
        ReduceToDevice0Launcher<float, float>(this->data);
    }
    else if (data[0].scalar_type() == kFloat64)
    {
        ReduceToDevice0Launcher<double, double>(this->data);
    }else
    {
        CHECK(false);
    }
    CUDA_SYNC_CHECK_ERROR();
}

void MultiDeviceTensor::copy_parameters_from_main_to_others_uva()
{
    for (int i = 1; i < data.size(); ++i)
    {
        int64_t numel = data[0].numel();
        DeviceGuard dg(data[i].device());

        if (data[0].scalar_type() == kFloat32)
        {
            auto src = data[0].data_ptr<float>();
            auto dst = data[i].data_ptr<float>();
            Simplecopy<<<iDivUp(numel, 128), 128, 0, getCurrentCUDAStream()>>>(src, dst, numel);
        }
        else if (data[0].scalar_type() == kFloat64)
        {
            auto src = data[0].data_ptr<double>();
            auto dst = data[i].data_ptr<double>();
            Simplecopy<<<iDivUp(numel, 128), 128, 0, getCurrentCUDAStream()>>>(src, dst, numel);
        }else
        {
            CHECK(false);
        }
        CUDA_SYNC_CHECK_ERROR();
    }
}


}  // namespace cuda
}  // namespace tinytorch

#endif