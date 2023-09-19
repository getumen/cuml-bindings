#include "cuml4c/device_resource_handle.h"

#include <raft/core/handle.hpp>

#include <memory>

namespace cuml4c
{
    struct DeviceResource
    {
        __host__ explicit DeviceResource(std::unique_ptr<raft::handle_t> handle) : handle(std::move(handle)) {}

        std::unique_ptr<raft::handle_t> handle;
    };
}
