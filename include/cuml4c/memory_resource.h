#ifdef __cplusplus
#define EXTERN_C extern "C"
#include <cstddef>
#else
#define EXTERN_C
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#endif

typedef void *DeviceMemoryResource;

EXTERN_C int UsePoolMemoryResource(
    size_t initial_pool_size,
    size_t maximum_pool_size,
    DeviceMemoryResource *resource);

EXTERN_C int UseBinningMemoryResource(
    int8_t min_size_exponent,
    int8_t max_size_exponent,
    DeviceMemoryResource *resource);

EXTERN_C int UseArenaMemoryResource(
    DeviceMemoryResource *resource);

EXTERN_C int ResetMemoryResource(
    DeviceMemoryResource resource,
    int resource_type);
