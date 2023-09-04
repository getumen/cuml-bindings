#pragma once

#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>

namespace cuml4c
{
    namespace handle_utils
    {

        void initializeHandle(raft::handle_t &handle,
                              rmm::cuda_stream_view stream_view = {});

    } // namespace handle_utils
} // namespace cuml4c
