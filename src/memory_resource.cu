#include "cuml4c/memory_resource.h"

#include <memory>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>

__host__ int UsePoolMemoryResource(
    size_t initial_pool_size,
    size_t maximum_pool_size,
    DeviceMemoryResource *resource)
{
    auto mr = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>>(
        rmm::mr::get_current_device_resource(),
        thrust::optional<size_t>(initial_pool_size),
        thrust::optional<size_t>(maximum_pool_size));

    rmm::mr::set_current_device_resource(mr.get());

    *resource = mr.release();

    return 0;
}

__host__ int UseBinningMemoryResource(
    int8_t min_size_exponent,
    int8_t max_size_exponent,
    DeviceMemoryResource *resource)
{
    auto mr = std::make_unique<rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource>>(
        rmm::mr::get_current_device_resource(),
        min_size_exponent,
        max_size_exponent);

    rmm::mr::set_current_device_resource(mr.get());

    *resource = mr.release();

    return 0;
}

__host__ int UseArenaMemoryResource(
    DeviceMemoryResource *resource)
{
    auto mr = std::make_unique<rmm::mr::arena_memory_resource<rmm::mr::device_memory_resource>>(
        rmm::mr::get_current_device_resource());

    rmm::mr::set_current_device_resource(mr.get());

    *resource = mr.release();
    return 0;
}

__host__ int ResetMemoryResource(
    DeviceMemoryResource resource,
    int resource_type)
{
    switch (resource_type)
    {
    case 0:
        delete static_cast<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> *>(resource);
        break;
    case 1:
        delete static_cast<rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource> *>(resource);
        break;
    case 2:
        delete static_cast<rmm::mr::arena_memory_resource<rmm::mr::device_memory_resource> *>(resource);
        break;
    default:
        return 1;
    }

    rmm::mr::set_current_device_resource(rmm::mr::detail::initial_resource());
    return 0;
}