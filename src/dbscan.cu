#include "async_utils.cuh"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "preprocessor.h"
#include "stream_allocator.h"
#include "cuml4c/dbscan.h"

#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/cluster/dbscan.hpp>

#include <memory>

__host__ int DbscanFit(
    const float *x,
    size_t num_row,
    size_t num_col,
    int min_pts,
    double eps,
    int metric,
    size_t max_bytes_per_batch,
    int verbosity,
    int *out)
{

    const size_t value_length = num_row * num_col;

    auto stream_view = cuml4c::stream_allocator::getOrCreateStream();
    raft::handle_t handle;
    cuml4c::handle_utils::initializeHandle(handle, stream_view.value());

    // dbscan input data
    thrust::device_vector<float> d_src_data(value_length);

    // dbscan output data
    thrust::device_vector<int> d_labels(num_row);

    // TODO: async copy
    thrust::copy(
        x,
        x + value_length,
        d_src_data.begin());

    ML::Dbscan::fit(handle, /*input=*/d_src_data.data().get(),
                    /*n_rows=*/num_row,
                    /*n_cols=*/num_col,
                    eps,
                    min_pts,
                    /*metric=*/static_cast<raft::distance::DistanceType>(metric),
                    /*labels=*/d_labels.data().get(),
                    /*core_sample_indices=*/nullptr,
                    max_bytes_per_batch,
                    /*verbosity=*/verbosity,
                    /*opg=*/false);

    // TODO: async copy
    thrust::copy(
        d_labels.begin(),
        d_labels.end(),
        out);

    return 0;
}
