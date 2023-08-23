#pragma once

#include <memory>

namespace raft
{
    namespace mr
    {
        namespace device
        {

            class allocator;

        } // namespace device
    }     // namespace mr
} // namespace raft

namespace cuml4c
{

    std::shared_ptr<raft::mr::device::allocator> getDeviceAllocator();

} // namespace cuml4c
