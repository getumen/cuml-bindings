#include "cuda_utils.h"
#include "handle_utils.h"
#include "preprocessor.h"
#include "stream_allocator.h"
#include "device_vector_utils.h"
#include "cuml4c/device_vector.h"
#include "cuml4c/agglomerative_clustering.h"

#include <thrust/device_vector.h>
#include <cuml/cluster/linkage.hpp>

#include <memory>

__host__ int AgglomerativeClusteringFit(
    DeviceVectorHandleFloat device_x,
    size_t num_row,
    size_t num_col,
    bool pairwise_conn,
    int metric,
    int n_neighbors,
    int init_n_clusters,
    int *n_clusters,
    DeviceVectorHandleInt *device_labels,
    DeviceVectorHandleInt *device_children)
{
    auto d_x = static_cast<cuml4c::DeviceVector<float> *>(device_x);

    auto d_labels = std::make_unique<thrust::device_vector<int>>(num_row);
    auto d_children = std::make_unique<thrust::device_vector<int>>((num_row - 1) * 2);

    auto stream_view = cuml4c::stream_allocator::getOrCreateStream();
    raft::handle_t handle;
    cuml4c::handle_utils::initializeHandle(handle, stream_view.value());

    // single-linkage hierarchical clustering output
    auto out = std::make_unique<raft::hierarchy::linkage_output<int, float>>();
    out->labels = d_labels->data().get();
    out->children = d_children->data().get();

    if (pairwise_conn)
    {
        ML::single_linkage_pairwise(
            handle,
            /*X=*/d_x->vector->data().get(),
            /*m=*/num_row,
            /*n=*/num_col,
            /*out=*/out.get(),
            /*metric=*/static_cast<raft::distance::DistanceType>(metric),
            init_n_clusters);
    }
    else
    {
        ML::single_linkage_neighbors(
            handle,
            /*X=*/d_x->vector->data().get(),
            /*m=*/num_row,
            /*n=*/num_col,
            /*out=*/out.get(),
            /*metric=*/static_cast<raft::distance::DistanceType>(metric),
            /*c=*/n_neighbors,
            init_n_clusters);
    }

    auto p_labels = std::make_unique<cuml4c::DeviceVector<int>>(std::move(d_labels));
    *device_labels = static_cast<DeviceVectorHandleInt>(p_labels.release());
    auto p_children = std::make_unique<cuml4c::DeviceVector<int>>(std::move(d_children));
    *device_children = static_cast<DeviceVectorHandleInt>(p_children.release());

    *n_clusters = out->n_clusters;

    return 0;
}
