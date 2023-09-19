#include "cuml4c/dbscan.h"
#include "device_resource_handle.cuh"

#include <thrust/copy.h>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <cuml/cluster/dbscan.hpp>

#include <memory>

__host__ int DbscanFit(
    const DeviceResourceHandle handle,
    const float *x,
    size_t num_row,
    size_t num_col,
    int min_pts,
    double eps,
    int metric,
    size_t max_bytes_per_batch,
    int verbosity,
    int *labels)
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

    ML::Dbscan::fit(*handle_p->handle,
                    /*input=*/d_x.begin(),
                    /*n_rows=*/num_row,
                    /*n_cols=*/num_col,
                    eps,
                    min_pts,
                    /*metric=*/static_cast<raft::distance::DistanceType>(metric),
                    /*labels=*/d_labels.begin(),
                    /*core_sample_indices=*/nullptr,
                    max_bytes_per_batch,
                    /*verbosity=*/verbosity,
                    /*opg=*/false);

    raft::update_host(labels,
                      d_labels.begin(),
                      d_labels.size(),
                      handle_p->handle->get_stream());

    handle_p->handle->get_stream().synchronize();

    return 0;
}
