#include "cuml4c/agglomerative_clustering.h"
#include "device_resource_handle.cuh"

#include <thrust/copy.h>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <cuml/cluster/linkage.hpp>

#include <memory>

__host__ int AgglomerativeClusteringFit(
    const DeviceResourceHandle handle,
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
    auto handle_p = static_cast<cuml4c::DeviceResource *>(handle);

    auto d_x = rmm::device_uvector<float>(
        num_col * num_row,
        handle_p->handle->get_stream());

    raft::update_device(d_x.data(),
                        x,
                        num_col * num_row,
                        handle_p->handle->get_stream());

    auto d_labels = rmm::device_uvector<int>(
        num_row,
        handle_p->handle->get_stream());

    auto d_children = rmm::device_uvector<int>(
        (num_row - 1) * 2,
        handle_p->handle->get_stream());

    // single-linkage hierarchical clustering output
    auto out = std::make_unique<raft::hierarchy::linkage_output<int>>();
    out->labels = d_labels.begin();
    out->children = d_children.begin();

    if (pairwise_conn)
    {
        ML::single_linkage_pairwise(
            *handle_p->handle,
            /*X=*/d_x.begin(),
            /*m=*/num_row,
            /*n=*/num_col,
            /*out=*/out.get(),
            /*metric=*/static_cast<raft::distance::DistanceType>(metric),
            init_n_clusters);
    }
    else
    {
        ML::single_linkage_neighbors(
            *handle_p->handle,
            /*X=*/d_x.begin(),
            /*m=*/num_row,
            /*n=*/num_col,
            /*out=*/out.get(),
            /*metric=*/static_cast<raft::distance::DistanceType>(metric),
            /*c=*/n_neighbors,
            init_n_clusters);
    }
    *n_clusters = out->n_clusters;

    raft::update_host(labels,
                      d_labels.begin(),
                      d_labels.size(),
                      handle_p->handle->get_stream());

    raft::update_host(children,
                      d_children.begin(),
                      d_children.size(),
                      handle_p->handle->get_stream());

    handle_p->handle->get_stream().synchronize();

    return 0;
}
