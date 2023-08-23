#include "async_utils.cuh"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "preprocessor.h"
#include "stream_allocator.h"
#include "cuml4c/kmeans.h"

#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/cluster/kmeans.hpp>

#include <memory>

__host__ int KmeansFit(
    const float *x,
    int num_row,
    int num_col,
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
    int *n_iter)
{

    ML::kmeans::KMeansParams params;
    params.n_clusters = k;
    params.max_iter = max_iters;
    if (tol > 0)
    {
        params.tol = tol;
        params.inertia_check = true;
    }

    params.init = static_cast<ML::kmeans::KMeansParams::InitMethod>(init_method);
    params.seed = seed;
    params.verbosity = verbosity;
    params.metric = metric;

    auto stream_view = cuml4c::stream_allocator::getOrCreateStream();
    raft::handle_t handle;
    cuml4c::handle_utils::initializeHandle(handle, stream_view.value());

    // kmeans input data
    const auto value_length = num_row * num_col;

    auto const n_centroid_values = params.n_clusters * num_col;
    thrust::device_vector<float> d_src_data(value_length);
    // TODO: async copy
    thrust::copy(
        x,
        x + value_length,
        d_src_data.begin());

    thrust::device_vector<float> d_sample_weight(num_row);
    if (sample_weight != nullptr)
    {
        thrust::copy(
            sample_weight,
            sample_weight + num_row,
            d_sample_weight.begin());
    }
    else
    {
        thrust::fill(
            d_sample_weight.begin(),
            d_sample_weight.end(),
            1.0f);
    }

    // kmeans outputs
    thrust::device_vector<float> d_pred_centroids(n_centroid_values);
    if (params.init == ML::kmeans::KMeansParams::InitMethod::Array)
    {
        // TODO: async copy
        thrust::copy(
            centroids,
            centroids + n_centroid_values,
            d_pred_centroids.begin());
    }
    thrust::device_vector<int> d_pred_labels(num_row);

    ML::kmeans::fit_predict(
        handle,
        params,
        d_src_data.data().get(),
        num_row,
        num_col,
        d_sample_weight.data().get(),
        d_pred_centroids.data().get(),
        d_pred_labels.data().get(),
        *inertia,
        *n_iter);

    // TODO: async copy
    thrust::copy(
        d_pred_labels.begin(),
        d_pred_labels.end(),
        labels);

    // TODO: async copy
    thrust::copy(
        d_pred_centroids.begin(),
        d_pred_centroids.end(),
        centroids);

    return 0;
}
