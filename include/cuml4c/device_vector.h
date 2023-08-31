#ifdef __cplusplus
#define EXTERN_C extern "C"
#include <cstddef>
#else
#define EXTERN_C
#include <stdbool.h>
#include <stdio.h>
#endif

typedef void *DeviceVectorHandleFloat;
typedef void *DeviceVectorHandleInt;

EXTERN_C int DeviceVectorToHostVectorFloat(
    DeviceVectorHandleFloat device,
    float *out);

EXTERN_C int HostVectorToDeviceVectorFloat(
    const float *host,
    size_t size,
    DeviceVectorHandleFloat *out);

EXTERN_C int DeviceVectorFloatGetSize(
    DeviceVectorHandleFloat device,
    size_t *out);

EXTERN_C int DeviceVectorFloatFree(
    DeviceVectorHandleFloat device);

EXTERN_C int DeviceVectorToHostVectorInt(
    DeviceVectorHandleInt device,
    int *out);

EXTERN_C int HostVectorToDeviceVectorInt(
    const int *host,
    size_t size,
    DeviceVectorHandleInt *out);

EXTERN_C int DeviceVectorIntGetSize(
    DeviceVectorHandleInt device,
    size_t *out);

EXTERN_C int DeviceVectorIntFree(
    DeviceVectorHandleInt device);
