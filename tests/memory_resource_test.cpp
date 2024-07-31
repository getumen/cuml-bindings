#include <gtest/gtest.h>

#include "cuml4c/device_resource_handle.h"
#include "cuml4c/memory_resource.h"

TEST(MemoryResourceTest, TestUsePoolMemoryResource)
{
    DeviceResourceHandle device_resource_handle;
    CreateDeviceResourceHandle(&device_resource_handle);

    DeviceMemoryResource mr;
    UsePoolMemoryResource(1024 * 1024, 8 * 1024 * 1024, &mr);

    ResetMemoryResource(mr, 0);

    FreeDeviceResourceHandle(device_resource_handle);
}

TEST(MemoryResourceTest, TestUseBinningMemoryResource)
{
    DeviceResourceHandle device_resource_handle;
    CreateDeviceResourceHandle(&device_resource_handle);

    DeviceMemoryResource mr;
    UseBinningMemoryResource(3, 10, &mr);

    ResetMemoryResource(mr, 1);

    FreeDeviceResourceHandle(device_resource_handle);
}

TEST(MemoryResourceTest, TestUseArenaMemoryResource)
{
    DeviceResourceHandle device_resource_handle;
    CreateDeviceResourceHandle(&device_resource_handle);

    DeviceMemoryResource mr;
    UseArenaMemoryResource(&mr, 1024 * 1024);

    ResetMemoryResource(mr, 2);

    FreeDeviceResourceHandle(device_resource_handle);
}