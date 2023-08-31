#ifdef __cplusplus
#define EXTERN_C extern "C"
#include <cstddef>
#else
#define EXTERN_C
#include <stdbool.h>
#include <stdio.h>
#endif

#include "cuml4c/device_vector.h"

typedef void *FILModelHandle;

EXTERN_C int FILLoadModel(
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
    FILModelHandle handle);

EXTERN_C int FILGetNumClasses(
    FILModelHandle model,
    size_t *out);

EXTERN_C int FILPredict(
    FILModelHandle model,
    DeviceVectorHandleFloat device_x,
    size_t num_row,
    bool output_class_probabilities,
    DeviceVectorHandleFloat device_preds);
