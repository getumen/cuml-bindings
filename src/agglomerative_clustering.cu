#include "async_utils.cuh"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "preprocessor.h"
#include "stream_allocator.h"
#include "cuml4c/agglomerative_clustering.h"

#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/cluster/linkage.hpp>

#include <memory>

__host__ int AgglomerativeClusteringFit(
    const float *x,
    size_t num_row,
    size_t num_col,
    bool pairwise_conn,
    int metric,
    int n_neighbors,
    int init_n_clusters,
    int *n_clusters,
    int *labels,
    int *children)
{

    const auto value_length = num_row * num_col;

    auto stream_view = cuml4c::stream_allocator::getOrCreateStream();
    raft::handle_t handle;
    cuml4c::handle_utils::initializeHandle(handle, stream_view.value());

    // single-linkage hierarchical clustering input
    thrust::device_vector<float> d_x(value_length);
    thrust::copy(x, x + value_length, d_x.begin());

    // single-linkage hierarchical clustering output
    auto out = std::make_unique<raft::hierarchy::linkage_output<int, float>>();
    thrust::device_vector<int> d_labels(num_row);
    thrust::device_vector<int> d_children((num_row - 1) * 2);
    out->labels = d_labels.data().get();
    out->children = d_children.data().get();

    if (pairwise_conn)
    {
        ML::single_linkage_pairwise(
            handle,
            /*X=*/d_x.data().get(),
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
            /*X=*/d_x.data().get(),
            /*m=*/num_row,
            /*n=*/num_col,
            /*out=*/out.get(),
            /*metric=*/static_cast<raft::distance::DistanceType>(metric),
            /*c=*/n_neighbors,
            init_n_clusters);
    }

    thrust::copy(
        d_labels.begin(),
        d_labels.end(),
        labels);
    thrust::copy(
        d_children.begin(),
        d_children.end(),
        children);

    *n_clusters = out->n_clusters;

    return 0;
}
