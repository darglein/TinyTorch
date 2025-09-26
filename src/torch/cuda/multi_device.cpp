#include "torch/tiny_torch_config.h"

#ifdef TT_HAS_CUDA

#    include "torch/core/graph.h"

#    include "multi_device.h"
#    include "ops_impl_cuda_helper.h"
#    include "torch/core/ops/ops_operators.h"
#    include "torch/core/ops/ops_tensor_creation.h"

namespace tinytorch
{
namespace cuda
{
static bool has_p2p_copy = false;
bool HasP2PCopy()
{
    return has_p2p_copy;
}


bool IsCudaPeerToPeerAvailable(Device device0, Device device1)
{
    CHECK_EQ(device0.type(), kCUDA);
    CHECK_EQ(device1.type(), kCUDA);
    CHECK_NE(device0.index(), device1.index());

    int canAccessPeer01, canAccessPeer10;
    TT_CHECK_CUDA_ERROR(cudaDeviceCanAccessPeer(&canAccessPeer01, device0.index(), device1.index()));
    TT_CHECK_CUDA_ERROR(cudaDeviceCanAccessPeer(&canAccessPeer10, device1.index(), device0.index()));

    return canAccessPeer01 && canAccessPeer10;
}

bool EnableCudaPeerToPeer(Device device0, Device device1)
{
    if (!IsCudaPeerToPeerAvailable(device0, device1))
    {
        return false;
    }

    CHECK_EQ(device0.type(), kCUDA);
    CHECK_EQ(device1.type(), kCUDA);
    CHECK_NE(device0.index(), device1.index());

    {
        cuda::DeviceGuard guard(device0);
        auto error = cudaDeviceEnablePeerAccess(device1.index(), 0);
        if (error != cudaErrorPeerAccessAlreadyEnabled)
        {
            TT_CHECK_CUDA_ERROR(error);
        }
    }
    {
        cuda::DeviceGuard guard(device1);
        auto error = cudaDeviceEnablePeerAccess(device0.index(), 0);
        if (error != cudaErrorPeerAccessAlreadyEnabled)
        {
            TT_CHECK_CUDA_ERROR(error);
        }
    }
    return true;
}

void DisableCudaPeerToPeer(Device device0, Device device1)
{
    if (!IsCudaPeerToPeerAvailable(device0, device1))
    {
        return;
    }

    CHECK_EQ(device0.type(), kCUDA);
    CHECK_EQ(device1.type(), kCUDA);
    CHECK_NE(device0.index(), device1.index());

    {
        cuda::DeviceGuard guard(device0);
        TT_CHECK_CUDA_ERROR(cudaDeviceDisablePeerAccess(device1.index()));
    }
    {
        cuda::DeviceGuard guard(device1);
        TT_CHECK_CUDA_ERROR(cudaDeviceDisablePeerAccess(device0.index()));
    }
}
bool EnableCudaPeerToPeer(const std::vector<Device>& devices)
{
    bool enabled = true;
    for (int i = 0; i < devices.size(); ++i)
    {
        for (int j = i + 1; j < devices.size(); ++j)
        {
            enabled &= EnableCudaPeerToPeer(devices[i], devices[j]);
        }
    }
    has_p2p_copy = enabled;
    return enabled;
}
void DisableCudaPeerToPeer(const std::vector<Device>& devices)
{
    if (devices.size() >= 2)
    {
        for (int i = 1; i < devices.size(); ++i)
        {
            DisableCudaPeerToPeer(devices[0], devices[i]);
        }
    }
    has_p2p_copy = false;
}

void MultiDeviceTensor::SetMain(Tensor t)
{
    Main() = t;
    if (t.defined())
    {
        devices.front() = t.device();
    }

    for (int local_device_id = 1; local_device_id < devices.size(); ++local_device_id)
    {
        auto d    = devices[local_device_id];
        auto& dst = data[local_device_id];
        if (!dst.defined())
        {
            if (t.defined())
            {
                dst = empty_like(t, t.options().device(d));
            }
            else
            {
                dst = {};
            }
        }
    }
}

void MultiDeviceTensor::MainToCPU()
{
    ToCPUSingle(0);
}


void MultiDeviceTensor::AllToCPU()
{
    for (int i = 0; i < data.size(); ++i)
    {
        ToCPUSingle(i);
    }
}


void MultiDeviceTensor::ToCPUSingle(int i)
{
    NoGradGuard ngg;
    if (!data[i].defined())
    {
        cpu_data[i] = {};
        return;
    }
    if (!cpu_data[i].defined())
    {
        cpu_data[i] = empty_like(data[i], data[i].options().device(kCPU).pinned_memory(true));
    }
    cpu_data[i].copy_(data[i], true);
}


void MultiDeviceTensor::ReduceGradientSumToMainUVA()
{
    for (int i = 1; i < cpu_data.size(); ++i)
    {
        CHECK(data[0].is_uva());
        CHECK(data[i].is_uva());
        data[0].mutable_grad() += data[i].mutable_grad();
    }
}
void MultiDeviceTensor::ReduceSumToMainUVA()
{
    for (int i = 1; i < cpu_data.size(); ++i)
    {
        CHECK(data[0].is_uva());
        CHECK(data[i].is_uva());
        data[0] += data[i];
    }
}
void MultiDeviceTensor::ReduceSumOnCPUToMain()
{
    for (int i = 1; i < cpu_data.size(); ++i)
    {
        ReduceSumCPUSingle(0, i);
    }
}


void MultiDeviceTensor::ReduceSumCPUSingle(int target, int src)
{
    if (cpu_data[target].scalar_type() == kFloat32 && cpu_data[target].is_contiguous())
    {
        float* target_ptr = cpu_data[target].data_ptr<float>();
        float* src_ptr    = cpu_data[src].data_ptr<float>();
        int64_t N         = cpu_data[target].numel();
#    pragma omp simd
        for (int64_t k = 0; k < N; ++k)
        {
            target_ptr[k] += src_ptr[k];
        }
    }
    else
    {
        NoGradGuard ngg;
        cpu_data[target] += cpu_data[src];
    }
}



void MultiDeviceTensor::MainCPUToOthers(cudaEvent_t wait_event, bool include_to_main_gpu)
{
    int start_id = include_to_main_gpu ? 0 : 1;
    for (int local_device_id = start_id; local_device_id < devices.size(); ++local_device_id)
    {
        auto d    = devices[local_device_id];
        auto& dst = data[local_device_id];
        cuda::DeviceGuard dg(d);

        if (!cpu_data.front().defined())
        {
            dst = {};
            continue;
        }

        if (wait_event)
        {
            TT_CHECK_CUDA_ERROR(cudaStreamWaitEvent(getCurrentCUDAStream(), wait_event));
        }

        dst.copy_(cpu_data.front(), true);
    }
}

void MultiDeviceTensor::MainCPUToMainGPU()
{
    cuda::DeviceGuard dg(devices[0]);
    data[0].copy_(cpu_data[0], true);
}



void MultiDeviceTensor::SetMainAndCopyToOthers(Tensor t)
{
    SetMain(t);
    MainToOthers();
}

void MultiDeviceTensor::MainToOthers()
{
    if (devices.size() == 1)
    {
        return;
    }

    if (devices.size() == 2 && Main().is_uva() && data[1].defined() && data[1].is_uva())
    {
        // direct uva copy
        NoGradGuard ngg;
        data[1].copy_(Main(), true);
        return;
    }

    MainToCPU();

    auto on_cpu_event = getNextEvent();
    cudaEventRecord(on_cpu_event, getCurrentCUDAStream());

    MainCPUToOthers(on_cpu_event, false);
}

MultiDeviceTensor MultiDeviceTensor::slice(int64_t d, int64_t start, int64_t end) const
{
    MultiDeviceTensor result = *this;
    for (int i = 0; i < data.size(); ++i)
    {
        if (cpu_data[i].defined())
        {
            result.cpu_data[i] = cpu_data[i].slice(d, start, end);
        }
        if (data[i].defined())
        {
            result.data[i] = data[i].slice(d, start, end);
        }
    }
    return result;
}
MultiDeviceTensor MultiDeviceTensor::view(const SizeType& sizes) const
{
    MultiDeviceTensor result = *this;
    for (int i = 0; i < data.size(); ++i)
    {
        if (cpu_data[i].defined())
        {
            result.cpu_data[i] = cpu_data[i].view(sizes);
        }
        if (data[i].defined())
        {
            result.data[i] = data[i].view(sizes);
        }
    }
    return result;
}
MultiDeviceTensor MultiDeviceTensor::reinterpret_view(ScalarType new_scalar_type) const
{
    MultiDeviceTensor result = *this;
    for (int i = 0; i < data.size(); ++i)
    {
        if (cpu_data[i].defined())
        {
            result.cpu_data[i] = cpu_data[i].reinterpret_view(new_scalar_type);
        }
        if (data[i].defined())
        {
            result.data[i] = data[i].reinterpret_view(new_scalar_type);
        }
    }
    return result;
}
void MultiDeviceTensor::AllocateFullCPU()
{
    for (int i = 0; i < data.size(); ++i)
    {
        if (!data[i].defined())
        {
            cpu_data[i] = {};
            continue;
        }
        if (!cpu_data[i].defined())
        {
            cpu_data[i] = empty_like(data[i], data[i].options().device(kCPU).pinned_memory(true));
        }
    }
}

}  // namespace cuda
}  // namespace tinytorch

#endif