#pragma once

#include "torch/cuda/tt_cuda.h"

namespace tinytorch
{
namespace cuda
{

TINYTORCH_API bool HasP2PCopy();

TINYTORCH_API bool IsCudaPeerToPeerAvailable(Device device0, Device device1);

// Must be called for all active devices to enable torch-p2p copies
TINYTORCH_API bool EnableCudaPeerToPeer(const std::vector<Device>& devices);
TINYTORCH_API void DisableCudaPeerToPeer(const std::vector<Device>& devices);



struct TINYTORCH_API MultiDeviceTensor
{
    std::vector<Device> devices;
    std::vector<Tensor> data;
    Tensor cpu_data;

    // Init with undefined tensor
    MultiDeviceTensor()
    {
        // data.push_back({});
        // devices.push_back({});
    }
    MultiDeviceTensor(Tensor d, std::vector<Device> _devices) : devices(_devices)
    {
        CHECK(d.defined());
        CHECK(!devices.empty());
        CHECK_EQ(d.device() , devices.front());
        data.resize(devices.size());
        SetMainAndCopyToOthers(d);
    }
    //    MultiDeviceTensor(std::vector<Device> _devices) : MultiDeviceTensor(Tensor(), _devices) {}

    Tensor* operator->()
    {
        CHECK(!data.empty());
        return &data.front();
    }

    // implicit cast to main_device tensor, to make use of existing functions
    operator Tensor&()
    {
        CHECK(!data.empty());
        return data.front();
    }
    operator const Tensor&() const
    {
        CHECK(!data.empty());
        return data.front();
    }


    Tensor& Main()
    {
        CHECK(!data.empty());
        return data.front();
    }

    bool Initialized() const { return !data.empty(); }

    void SetMain(Tensor t);
    void MainToCPU();
    void CPUToOthers(cudaEvent_t wait_event);

    void zero_()
    {
        for (auto& d : data)
        {
            d.zero_();
        }
    }

    MultiDeviceTensor slice(int64_t d, int64_t start, int64_t end) const;


    Tensor operator[](Device d)
    {
        for (int i = 0; i < devices.size(); ++i)
        {
            if (devices[i] == d)
            {
                return data[i];
            }
        }
        CHECK(false) << "Device " << d << " not found!";
        return {};
    }

    void SetMainAndCopyToOthers(Tensor t);
    void MainToOthers();
};


}  // namespace cuda
}  // namespace tinytorch