#ifdef __cplusplus
#define EXTERN_C extern "C"
#include <cstddef>
#else
#define EXTERN_C
#include <stdbool.h>
#include <stdio.h>
#endif

typedef void *DeviceResourceHandle;

EXTERN_C int CreateDeviceResourceHandle(DeviceResourceHandle *handle);

EXTERN_C int FreeDeviceResourceHandle(DeviceResourceHandle handle);
