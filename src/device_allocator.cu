#include "device_allocator.h"

#include <raft/mr/device/allocator.hpp>

namespace
{

    auto const kDefaultDeviceAllocator =
        std::make_shared<raft::mr::device::default_allocator>();

} // namespace

namespace cuml4c
{

    __host__ std::shared_ptr<raft::mr::device::allocator> getDeviceAllocator()
    {
        return kDefaultDeviceAllocator;
    }

} // namespace cuml4c
