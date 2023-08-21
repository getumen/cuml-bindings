#pragma once

#include "cuda_utils.h"
#include "preprocessor.h"
#include "unique_marker.cuh"

#include <thrust/async/copy.h>
#include <thrust/system/cuda/future.h>

#include <utility>

namespace cuml4c
{

    // To ensure the correct async behavior, an `AsyncCopyCtx` object must be
    // destroyed after the stream associated with the copy operation is
    // synchronized, not before.
    struct AsyncCopyCtx
    {
        thrust::system::cuda::unique_eager_event event;
        unique_marker marker;
    };

    // perform a copy operation that is asynchronous with respect to the host
    // and synchronous with respect to the stream specified
    template <typename... Args>
    __host__ CUML4C_NODISCARD auto async_copy(cudaStream_t stream, Args &&...args)
    {
        auto e = thrust::async::copy(std::forward<Args>(args)...);
        auto &s = e.stream();
        unique_marker m;
        CUDA_RT_CALL(cudaEventRecord(m.get(), s.get()));
        CUDA_RT_CALL(cudaStreamWaitEvent(stream, m.get(), cudaEventWaitDefault));
        return AsyncCopyCtx{std::move(e), std::move(m)};
    }

} // namespace cuml4c
