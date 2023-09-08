#include "cuml4c/kmeans.h"

#include <thrust/copy.h>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <cuml/cluster/kmeans.hpp>

#include <memory>

__host__ int KmeansFit(
    const float *x,
    int num_row,
    int num_col,
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
    int *n_iter)
{

    auto handle = std::make_shared<raft::handle_t>();

    auto d_x = rmm::device_uvector<float>(
        num_col * num_row,
        handle->get_stream());

    raft::update_device(d_x.data(),
                        x,
                        num_col * num_row,
                        handle->get_stream());

    auto d_labels = rmm::device_uvector<int>(
        num_row,
        handle->get_stream());

    auto d_centroids = rmm::device_uvector<float>(
        k * num_col,
        handle->get_stream());

    ML::kmeans::KMeansParams params;
    params.n_clusters = k;
    params.max_iter = max_iters;
    if (tol > 0)
    {
        params.tol = tol;
        params.inertia_check = true;
    }

    params.init = static_cast<ML::kmeans::KMeansParams::InitMethod>(init_method);
    params.verbosity = verbosity;
    params.metric = static_cast<raft::distance::DistanceType>(metric);

    ML::kmeans::fit_predict(
        *handle,
        params,
        d_x.begin(),
        num_row,
        num_col,
        nullptr,
        d_centroids.begin(),
        d_labels.begin(),
        *inertia,
        *n_iter);

    raft::update_host(labels,
                      d_labels.begin(),
                      d_labels.size(),
                      handle->get_stream());

    raft::update_host(centroids,
                      d_centroids.begin(),
                      d_centroids.size(),
                      handle->get_stream());

    handle->get_stream().synchronize();

    return 0;
}
