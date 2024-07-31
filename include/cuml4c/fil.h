#ifdef __cplusplus
#define EXTERN_C extern "C"
#include <cstddef>
#else
#define EXTERN_C
#include <stdbool.h>
#include <stdio.h>
#endif

#include "cuml4c/device_resource_handle.h"

typedef void *FILModelHandle;

enum FILStatus
{
    FIL_SUCCESS = 0,
    FIL_FAIL_TO_LOAD_MODEL = 1,
    FIL_FAIL_TO_GET_NUM_CLASS = 2,
    FIL_FAIL_TO_GET_NUM_FEATURE = 3,
    FIL_INVALID_ARGUMENT = 4,
    FIL_FAIL_TO_FREE_MODEL = 5,
};

EXTERN_C int FILLoadModel(
    const DeviceResourceHandle handle,
    int model_type,
    const char *filename,
    int algo,
    bool classification,
    float threshold,
    int storage_type,
    int blocks_per_sm,
    int threads_per_tree,
    int n_items,
    FILModelHandle *out);

EXTERN_C int FILFreeModel(
    const DeviceResourceHandle handle,
    FILModelHandle model);

EXTERN_C int FILPredict(
    const DeviceResourceHandle handle,
    FILModelHandle model,
    const float *x,
    size_t num_row,
    bool output_class_probabilities,
    float *preds);
