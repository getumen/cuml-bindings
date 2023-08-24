#ifdef __cplusplus
#define EXTERN_C extern "C"
#include <cstddef>
#else
#define EXTERN_C
#include <stdbool.h>
#include <stdio.h>
#endif

EXTERN_C int KmeansFit(
    const float *x,
    size_t num_row,
    size_t num_col,
    const float *sample_weight,
    int k,
    int max_iters,
    double tol,
    int init_method,
    int metric,
    int seed,
    int verbosity,
    int *labels,
    float *centroids,
    float *inertia,
    int *n_iter);
