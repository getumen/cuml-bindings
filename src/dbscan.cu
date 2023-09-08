#include "cuml4c/dbscan.h"

#include <thrust/copy.h>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
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
    int *labels)
{
    auto handle = std::make_unique<raft::handle_t>();

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

    ML::Dbscan::fit(*handle,
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
                      handle->get_stream());

    handle->get_stream().synchronize();

    return 0;
}
