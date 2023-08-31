#include "device_vector_utils.h"

#include "cuml4c/device_vector.h"

#include <thrust/device_vector.h>

__host__ int DeviceVectorToHostVectorFloat(
    DeviceVectorHandleFloat device,
    float *out)
{
    auto d = static_cast<cuml4c::DeviceVector<float> *>(device);
    auto vector = d->vector.get();
    thrust::copy(vector->begin(), vector->end(), out);

    return 0;
}

__host__ int HostVectorToDeviceVectorFloat(
    const float *host,
    size_t size,
    DeviceVectorHandleFloat *out)
{
    auto d = std::make_unique<thrust::device_vector<float>>(size);

    thrust::copy(host, host + size, d->begin());

    auto p = std::make_unique<cuml4c::DeviceVector<float>>(std::move(d));

    *out = static_cast<DeviceVectorHandleFloat>(p.release());

    return 0;
}

__host__ int DeviceVectorFloatGetSize(
    DeviceVectorHandleFloat device,
    size_t *out)
{
    auto d = static_cast<cuml4c::DeviceVector<float> *>(device);
    auto vector = d->vector.get();
    *out = vector->size();

    return 0;
}

__host__ int DeviceVectorFloatFree(
    DeviceVectorHandleFloat device)
{
    delete static_cast<cuml4c::DeviceVector<int> *>(device);

    return 0;
}

__host__ int DeviceVectorToHostVectorInt(
    DeviceVectorHandleInt device,
    int *out)
{
    auto d = static_cast<cuml4c::DeviceVector<int> *>(device);
    auto vector = d->vector.get();
    thrust::copy(vector->begin(), vector->end(), out);

    return 0;
}

__host__ int HostVectorToDeviceVectorInt(
    const int *host,
    size_t size,
    DeviceVectorHandleInt *out)
{
    auto d = std::make_unique<thrust::device_vector<int>>(size);

    thrust::copy(host, host + size, d->begin());

    auto p = std::make_unique<cuml4c::DeviceVector<int>>(std::move(d));

    *out = static_cast<DeviceVectorHandleInt>(p.release());

    return 0;
}

__host__ int DeviceVectorIntGetSize(
    DeviceVectorHandleInt device,
    size_t *out)
{
    auto d = static_cast<cuml4c::DeviceVector<int> *>(device);
    auto vector = d->vector.get();
    *out = vector->size();

    return 0;
}

__host__ int DeviceVectorIntFree(
    DeviceVectorHandleInt device)
{
    delete static_cast<cuml4c::DeviceVector<int> *>(device);

    return 0;
}