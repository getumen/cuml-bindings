#ifdef __cplusplus
#define EXTERN_C extern "C"
#include <cstddef>
#else
#define EXTERN_C
#include <stdbool.h>
#include <stdio.h>
#endif

#include "cuml4c/device_vector.h"

EXTERN_C int AgglomerativeClusteringFit(
    DeviceVectorHandleFloat device_x,
    size_t num_row,
    size_t num_col,
    bool pairwise_conn,
    int metric,
    int n_neighbors,
    int init_n_clusters,
    int *n_clusters,
    DeviceVectorHandleInt device_labels,
    DeviceVectorHandleInt device_children);
