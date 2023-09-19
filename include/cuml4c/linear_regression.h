#ifdef __cplusplus
#define EXTERN_C extern "C"
#include <cstddef>
#else
#define EXTERN_C
#include <stdbool.h>
#include <stdio.h>
#endif

#include "cuml4c/device_resource_handle.h"

EXTERN_C int OlsFit(
    const DeviceResourceHandle handle,
    const float *x,
    size_t num_row,
    size_t num_col,
    const float *labels,
    bool fit_intercept,
    bool normalize,
    int algo,
    float *coef,
    float *intercept);

EXTERN_C int RidgeFit(
    const DeviceResourceHandle handle,
    const float *x,
    size_t num_row,
    size_t num_col,
    const float *labels,
    float *alpha,
    size_t n_alpha,
    bool fit_intercept,
    bool normalize,
    int algo,
    float *coef,
    float *intercept);

EXTERN_C int GemmPredict(
    const DeviceResourceHandle handle,
    const float *x,
    size_t num_row,
    size_t num_col,
    const float *coef,
    float intercept,
    float *preds);
