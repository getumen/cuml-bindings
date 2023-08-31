#include "cuda_utils.h"
#include "handle_utils.h"
#include "preprocessor.h"
#include "stream_allocator.h"
#include "device_vector_utils.h"
#include "cuml4c/device_vector.h"
#include "cuml4c/kmeans.h"

#include <thrust/device_vector.h>
#include <cuml/cluster/kmeans.hpp>

#include <memory>

// TODO: support initMethod == Array
__host__ int KmeansFit(
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
    int *n_iter)
{

    auto d_x = static_cast<cuml4c::DeviceVector<float> *>(device_x);

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

    auto const n_centroid_values = params.n_clusters * num_col;
    auto d_centroids = std::make_unique<thrust::device_vector<float>>(n_centroid_values);

    auto d_labels = std::make_unique<thrust::device_vector<int>>(num_row);

    if (device_sample_weight == nullptr)
    {
        ML::kmeans::fit_predict(
            handle,
            params,
            d_x->vector->data().get(),
            num_row,
            num_col,
            nullptr,
            d_centroids->data().get(),
            d_labels->data().get(),
            *inertia,
            *n_iter);
    }
    else
    {
        auto d_sample_weight = static_cast<cuml4c::DeviceVector<float> *>(device_sample_weight);

        ML::kmeans::fit_predict(
            handle,
            params,
            d_x->vector->data().get(),
            num_row,
            num_col,
            d_sample_weight->vector->data().get(),
            d_centroids->data().get(),
            d_labels->data().get(),
            *inertia,
            *n_iter);
    }

    auto p_labels = std::make_unique<cuml4c::DeviceVector<int>>(std::move(d_labels));
    *device_labels = static_cast<DeviceVectorHandleInt>(p_labels.release());

    auto p_centroids = std::make_unique<cuml4c::DeviceVector<float>>(std::move(d_centroids));
    *device_centroids = static_cast<DeviceVectorHandleFloat>(p_centroids.release());

    return 0;
}
