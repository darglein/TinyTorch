#include "multi_device.h"

#include "ops_impl_cuda_helper.h"

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
    CHECK_CUDA_ERROR(cudaDeviceCanAccessPeer(&canAccessPeer01, device0.index(), device1.index()));
    CHECK_CUDA_ERROR(cudaDeviceCanAccessPeer(&canAccessPeer10, device1.index(), device0.index()));

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
        CHECK_CUDA_ERROR(cudaDeviceDisablePeerAccess(device1.index()));
    }
    {
        cuda::DeviceGuard guard(device1);
        CHECK_CUDA_ERROR(cudaDeviceDisablePeerAccess(device0.index()));
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

}  // namespace cuda
}  // namespace tinytorch