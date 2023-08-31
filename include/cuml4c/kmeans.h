#ifdef __cplusplus
#define EXTERN_C extern "C"
#include <cstddef>
#else
#define EXTERN_C
#include <stdbool.h>
#include <stdio.h>
#endif

#include "cuml4c/device_vector.h"

EXTERN_C int KmeansFit(
    DeviceVectorHandleFloat device_x,
    int num_row,
    int num_col,
    DeviceVectorHandleFloat device_sample_weight,
    int k,
    int max_iters,
    double tol,
    int init_method,
    int metric,
    int seed,
    int verbosity,
    DeviceVectorHandleInt *device_labels,
    DeviceVectorHandleFloat *device_centroids,
    float *inertia,
    int *n_iter);
