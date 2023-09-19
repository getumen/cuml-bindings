#include "cuml4c/device_resource_handle.h"
#include "device_resource_handle.cuh"

#include <raft/core/handle.hpp>

#include <memory>

__host__ int CreateDeviceResourceHandle(DeviceResourceHandle *handle)
{
    auto raft_handle = std::make_unique<raft::handle_t>();

    auto p = std::make_unique<cuml4c::DeviceResource>(std::move(raft_handle));

    *handle = static_cast<DeviceResourceHandle>(p.release());

    return 0;
}

__host__ int FreeDeviceResourceHandle(DeviceResourceHandle handle)
{
    delete static_cast<cuml4c::DeviceResource *>(handle);
    return 0;
}
