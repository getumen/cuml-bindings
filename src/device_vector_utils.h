#include "cuml4c/device_vector.h"

#include <utility>

#include <thrust/device_vector.h>
#include <thrust/device_allocator.h>

namespace cuml4c
{
    template <typename T>
    struct DeviceVector
    {
        explicit DeviceVector(
            std::unique_ptr<thrust::device_vector<T, thrust::device_allocator<T>>> d_vector) : vector(std::move(d_vector))
        {
        }

        std::unique_ptr<thrust::device_vector<T>> vector;
    };
}