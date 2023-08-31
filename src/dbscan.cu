#include "cuda_utils.h"
#include "handle_utils.h"
#include "preprocessor.h"
#include "stream_allocator.h"
#include "device_vector_utils.h"
#include "cuml4c/device_vector.h"
#include "cuml4c/dbscan.h"

#include <thrust/device_vector.h>
#include <cuml/cluster/dbscan.hpp>

#include <memory>

__host__ int DbscanFit(
    DeviceVectorHandleFloat device_x,
    size_t num_row,
    size_t num_col,
    int min_pts,
    double eps,
    int metric,
    size_t max_bytes_per_batch,
    int verbosity,
    DeviceVectorHandleInt *device_labels)
{

    auto d_x = static_cast<cuml4c::DeviceVector<float> *>(device_x);

    // dbscan output data
    auto d_labels = std::make_unique<thrust::device_vector<int>>(num_row);

    auto stream_view = cuml4c::stream_allocator::getOrCreateStream();
    raft::handle_t handle;
    cuml4c::handle_utils::initializeHandle(handle, stream_view.value());

    ML::Dbscan::fit(handle,
                    /*input=*/d_x->vector->data().get(),
                    /*n_rows=*/num_row,
                    /*n_cols=*/num_col,
                    eps,
                    min_pts,
                    /*metric=*/static_cast<raft::distance::DistanceType>(metric),
                    /*labels=*/d_labels->data().get(),
                    /*core_sample_indices=*/nullptr,
                    max_bytes_per_batch,
                    /*verbosity=*/verbosity,
                    /*opg=*/false);

    auto p_labels = std::make_unique<cuml4c::DeviceVector<int>>(std::move(d_labels));
    *device_labels = static_cast<DeviceVectorHandleInt>(p_labels.release());

    return 0;
}
