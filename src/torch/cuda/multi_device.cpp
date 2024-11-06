#include "multi_device.h"

#include "ops_impl_cuda_helper.h"

namespace tinytorch
{
namespace cuda
{


bool IsCudaPeerToPeerAvailable(Device device0, Device device1)
{
    CHECK_EQ(device0.type(), kCUDA);
    CHECK_EQ(device1.type(), kCUDA);
    CHECK_NE(device0.index(), device1.index());

    int canAccessPeer01, canAccessPeer10;
    CHECK_CUDA_ERROR(cudaDeviceCanAccessPeer(&canAccessPeer01, device0.index(), device1.index()));
    CHECK_CUDA_ERROR(cudaDeviceCanAccessPeer(&canAccessPeer10, device1.index(), device0.index()));

    return canAccessPeer01 && canAccessPeer10;
}

bool EnableCudaPeerToPeer(Device device0, Device device1)
{
    if (!IsCudaPeerToPeerAvailable(device0, device1))
    {
#ifdef QRT_ENABLE_LOGGING
        std::cout << "CudaPeerToPeer not available " << device0 << " -> " << device1 << std::endl;
#endif
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
            CHECK_CUDA_ERROR(error);
        }
    }
    {
        cuda::DeviceGuard guard(device1);
        auto error = cudaDeviceEnablePeerAccess(device0.index(), 0);
        if (error != cudaErrorPeerAccessAlreadyEnabled)
        {
            CHECK_CUDA_ERROR(error);
        }
    }

#ifdef QRT_ENABLE_LOGGING
    std::cout << "CudaPeerToPeer enabled for device " << device0 << " -> " << device1 << std::endl;
#endif

    return true;
}

void DisableCudaPeerToPeer(Device device0, Device device1)
{
    CHECK_EQ(device0.type(), kCUDA);
    CHECK_EQ(device1.type(), kCUDA);
    CHECK_NE(device0.index(), device1.index());

    {
        cuda::DeviceGuard guard(device0);
        CHECK_CUDA_ERROR(cudaDeviceDisablePeerAccess(device1.index()));
    }
    {
        cuda::DeviceGuard guard(device1);
        CHECK_CUDA_ERROR(cudaDeviceDisablePeerAccess(device0.index()));
    }
}
bool EnableCudaPeerToPeerVec(const std::vector<Device>& devices)
{
    bool enabled = true;
    for (int i = 0; i < devices.size(); ++i)
    {
        for (int j = i + 1; j < devices.size(); ++j)
        {
            enabled &= EnableCudaPeerToPeer(devices[i], devices[j]);
        }
    }
    return enabled;
}
void DisableCudaPeerToPeerVec(const std::vector<Device>& devices)
{
    if (devices.size() >= 2)
    {
        for (int i = 1; i < devices.size(); ++i)
        {
            DisableCudaPeerToPeer(devices[0], devices[i]);
        }
    }
}

}  // namespace cuda
}  // namespace tinytorch