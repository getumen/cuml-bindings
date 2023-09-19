#include "cuml4c/linear_regression.h"
#include "device_resource_handle.cuh"

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <cuml/linear_model/glm.hpp>
#include <cuml/linear_model/qn.h>

#include <memory>

__host__ int OlsFit(
    const DeviceResourceHandle handle,
    const float *x,
    size_t num_row,
    size_t num_col,
    const float *labels,
    bool fit_intercept,
    bool normalize,
    int algo,
    float *coef,
    float *intercept)
{
    auto handle_p = static_cast<cuml4c::DeviceResource *>(handle);

    auto d_x = rmm::device_uvector<float>(
        num_col * num_row,
        handle_p->handle->get_stream());

    raft::update_device(d_x.data(),
                        x,
                        num_col * num_row,
                        handle_p->handle->get_stream());

    auto d_labels = rmm::device_uvector<float>(
        num_row,
        handle_p->handle->get_stream());

    raft::update_device(d_labels.data(),
                        labels,
                        num_row,
                        handle_p->handle->get_stream());

    auto d_coef = rmm::device_uvector<float>(
        num_col,
        handle_p->handle->get_stream());

    ML::GLM::olsFit(
        *handle_p->handle,
        d_x.begin(),
        int(num_row),
        int(num_col),
        d_labels.begin(),
        d_coef.begin(),
        intercept,
        fit_intercept,
        normalize,
        algo,
        nullptr);

    raft::update_host(coef,
                      d_coef.begin(),
                      d_coef.size(),
                      handle_p->handle->get_stream());

    handle_p->handle->get_stream().synchronize();

    return 0;
}

__host__ int RidgeFit(
    const DeviceResourceHandle handle,
    const float *x,
    size_t num_row,
    size_t num_col,
    const float *labels,
    float *alpha,
    size_t n_alpha,
    bool fit_intercept,
    bool normalize,
    int algo,
    float *coef,
    float *intercept)
{
    auto handle_p = static_cast<cuml4c::DeviceResource *>(handle);

    auto d_x = rmm::device_uvector<float>(
        num_col * num_row,
        handle_p->handle->get_stream());

    raft::update_device(d_x.data(),
                        x,
                        num_col * num_row,
                        handle_p->handle->get_stream());

    auto d_labels = rmm::device_uvector<float>(
        num_row,
        handle_p->handle->get_stream());

    raft::update_device(d_labels.data(),
                        labels,
                        num_row,
                        handle_p->handle->get_stream());

    auto d_coef = rmm::device_uvector<float>(
        num_col,
        handle_p->handle->get_stream());

    ML::GLM::ridgeFit(
        *handle_p->handle,
        d_x.begin(),
        num_row,
        num_col,
        d_labels.begin(),
        alpha,
        int(n_alpha),
        d_coef.begin(),
        intercept,
        fit_intercept,
        normalize,
        algo,
        nullptr);

    raft::update_host(coef,
                      d_coef.begin(),
                      d_coef.size(),
                      handle_p->handle->get_stream());

    handle_p->handle->get_stream().synchronize();

    return 0;
}

__host__ int GemmPredict(
    const DeviceResourceHandle handle,
    const float *x,
    size_t num_row,
    size_t num_col,
    const float *coef,
    float intercept,
    float *preds)
{
    auto handle_p = static_cast<cuml4c::DeviceResource *>(handle);

    auto d_x = rmm::device_uvector<float>(
        num_col * num_row,
        handle_p->handle->get_stream());

    raft::update_device(d_x.data(),
                        x,
                        num_col * num_row,
                        handle_p->handle->get_stream());

    auto d_coef = rmm::device_uvector<float>(
        num_col,
        handle_p->handle->get_stream());

    raft::update_device(d_coef.data(),
                        coef,
                        num_col,
                        handle_p->handle->get_stream());

    auto d_preds = rmm::device_uvector<float>(
        num_row,
        handle_p->handle->get_stream());

    ML::GLM::gemmPredict(
        *handle_p->handle,
        d_x.begin(),
        num_row,
        num_col,
        d_coef.begin(),
        intercept,
        d_preds.begin());

    raft::update_host(preds,
                      d_preds.begin(),
                      d_preds.size(),
                      handle_p->handle->get_stream());

    handle_p->handle->get_stream().synchronize();

    return 0;
}
