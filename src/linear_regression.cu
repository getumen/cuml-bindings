#include "stream_allocator.h"
#include "handle_utils.h"
#include "cuml4c/linear_regression.h"

#include <thrust/device_vector.h>
#include <cuml/linear_model/glm.hpp>

__host__ int OlsFit(
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
    auto stream_view = cuml4c::stream_allocator::getOrCreateStream();
    raft::handle_t handle;
    cuml4c::handle_utils::initializeHandle(handle, stream_view.value());

    const auto value_length = num_row * num_col;

    thrust::device_vector<float> d_src_data(value_length);

    thrust::copy(
        x,
        x + value_length,
        d_src_data.begin());

    thrust::device_vector<float> d_labels(num_row);

    thrust::copy(
        labels,
        labels + num_row,
        d_labels.begin());

    thrust::device_vector<float> d_coef(num_col);
    thrust::device_vector<float> d_intercept(1);

    ML::GLM::olsFit(
        handle,
        d_src_data.data().get(),
        int(num_row),
        int(num_col),
        d_labels.data().get(),
        d_coef.data().get(),
        d_intercept.data().get(),
        fit_intercept,
        normalize,
        algo);

    thrust::copy(
        d_coef.begin(),
        d_coef.end(),
        coef);

    thrust::copy(
        d_intercept.begin(),
        d_intercept.end(),
        intercept);

    return 0;
}

__host__ int RidgeFit(
    const float *x,
    size_t num_row,
    size_t num_col,
    const float *labels,
    const float *alpha,
    size_t n_alpha,
    bool fit_intercept,
    bool normalize,
    int algo,
    float *coef,
    float *intercept)
{
    auto stream_view = cuml4c::stream_allocator::getOrCreateStream();
    raft::handle_t handle;
    cuml4c::handle_utils::initializeHandle(handle, stream_view.value());

    const auto value_length = num_row * num_col;

    thrust::device_vector<float> d_src_data(value_length);

    thrust::copy(
        x,
        x + value_length,
        d_src_data.begin());

    thrust::device_vector<float> d_labels(num_row);

    thrust::copy(
        labels,
        labels + num_row,
        d_labels.begin());

    thrust::device_vector<float> d_alpha(n_alpha);

    thrust::copy(
        alpha,
        alpha + n_alpha,
        d_alpha.begin());

    thrust::device_vector<float> d_coef(num_col);
    thrust::device_vector<float> d_intercept(1);

    ML::GLM::ridgeFit(
        handle,
        d_src_data.data().get(),
        int(num_row),
        int(num_col),
        d_labels.data().get(),
        d_alpha.data().get(),
        int(n_alpha),
        d_coef.data().get(),
        d_intercept.data().get(),
        fit_intercept,
        normalize,
        algo);

    thrust::copy(
        d_coef.begin(),
        d_coef.end(),
        coef);

    thrust::copy(
        d_intercept.begin(),
        d_intercept.end(),
        intercept);

    return 0;
}

__host__ int GemmPredict(
    const float *x,
    size_t num_row,
    size_t num_col,
    const float *coef,
    float intercept,
    float *preds)
{
    auto stream_view = cuml4c::stream_allocator::getOrCreateStream();
    raft::handle_t handle;
    cuml4c::handle_utils::initializeHandle(handle, stream_view.value());

    thrust::device_vector<float> d_x(num_row * num_col);
    thrust::copy(
        x,
        x + num_col * num_row,
        d_x.begin());

    thrust::device_vector<float> d_coef(num_col);
    thrust::copy(
        coef,
        coef + num_col,
        d_coef.begin());

    thrust::device_vector<float> d_preds(num_row);

    ML::GLM::gemmPredict(
        handle,
        d_x.data().get(),
        int(num_row),
        int(num_col),
        d_coef.data().get(),
        intercept,
        d_preds.data().get());

    thrust::copy(
        d_preds.begin(),
        d_preds.end(),
        preds);

    return 0;
}

__host__ int QnFit(
    const float *x,
    size_t num_row,
    size_t num_col,
    bool X_col_major,
    const float *labels,
    size_t num_class,
    int loss_type,
    const float *sample_weight,
    bool fit_intercept,
    float l1,
    float l2,
    int max_iter,
    float grad_tol,
    float change_tol,
    int linesearch_max_iter,
    int lbfgs_memory,
    int verbosity,
    float *w0,
    float *f,
    int *num_iters)
{
    auto stream_view = cuml4c::stream_allocator::getOrCreateStream();
    raft::handle_t handle;
    cuml4c::handle_utils::initializeHandle(handle, stream_view.value());

    const auto value_length = num_row * num_col;

    thrust::device_vector<float> d_src_data(value_length);

    thrust::copy(
        x,
        x + value_length,
        d_src_data.begin());

    thrust::device_vector<float> d_labels(num_row);

    thrust::copy(
        labels,
        labels + num_row,
        d_labels.begin());

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

    thrust::device_vector<float> d_w0((num_col + (fit_intercept ? 1 : 0)) * num_class);

    ML::GLM::qnFit(
        handle,
        d_src_data.data().get(),
        X_col_major,
        d_labels.data().get(),
        int(num_row),
        int(num_col),
        int(num_class),
        fit_intercept,
        l1,
        l2,
        max_iter,
        grad_tol,
        change_tol,
        linesearch_max_iter,
        lbfgs_memory,
        verbosity,
        d_w0.data().get(),
        f,
        num_iters,
        loss_type,
        d_sample_weight.data().get());

    thrust::copy(
        d_w0.begin(),
        d_w0.end(),
        w0);

    return 0;
}

__host__ int QnFitSparse(
    const float *values,
    const int *indices,
    const int *header,
    size_t num_row,
    size_t num_col,
    size_t num_non_zero,
    const float *labels,
    size_t num_class,
    int loss_type,
    const float *sample_weight,
    bool fit_intercept,
    float l1,
    float l2,
    int max_iter,
    float grad_tol,
    float change_tol,
    int linesearch_max_iter,
    int lbfgs_memory,
    int verbosity,
    float *w0,
    float *f,
    int *num_iters)
{
    auto stream_view = cuml4c::stream_allocator::getOrCreateStream();
    raft::handle_t handle;
    cuml4c::handle_utils::initializeHandle(handle, stream_view.value());

    thrust::device_vector<float> d_values(num_non_zero);
    thrust::copy(
        values,
        values + num_non_zero,
        d_values.begin());

    thrust::device_vector<int> d_indices(num_non_zero);
    thrust::copy(
        indices,
        indices + num_non_zero,
        d_indices.begin());

    thrust::device_vector<int> d_header(num_row + 1);
    thrust::copy(
        header,
        header + num_row + 1,
        d_header.begin());

    thrust::device_vector<float> d_labels(num_row);

    thrust::copy(
        labels,
        labels + num_row,
        d_labels.begin());

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

    thrust::device_vector<float> d_w0((num_col + (fit_intercept ? 1 : 0)) * num_class);

    ML::GLM::qnFitSparse(
        handle,
        d_values.data().get(),
        d_indices.data().get(),
        d_header.data().get(),
        int(num_non_zero),
        d_labels.data().get(),
        int(num_row),
        int(num_col),
        int(num_class),
        fit_intercept,
        l1,
        l2,
        max_iter,
        grad_tol,
        change_tol,
        linesearch_max_iter,
        lbfgs_memory,
        verbosity,
        d_w0.data().get(),
        f,
        num_iters,
        loss_type,
        d_sample_weight.data().get());

    thrust::copy(
        d_w0.begin(),
        d_w0.end(),
        w0);

    return 0;
}

__host__ int QnDecisionFunction(
    const float *x,
    bool X_col_major,
    size_t num_row,
    size_t num_col,
    size_t num_class,
    bool fit_intercept,
    const float *params,
    int loss_type,
    float *preds)
{
    auto stream_view = cuml4c::stream_allocator::getOrCreateStream();
    raft::handle_t handle;
    cuml4c::handle_utils::initializeHandle(handle, stream_view.value());

    const auto value_length = num_row * num_col;

    thrust::device_vector<float> d_src_data(value_length);
    thrust::copy(
        x,
        x + value_length,
        d_src_data.begin());

    thrust::device_vector<float> d_params((num_col + (fit_intercept ? 1 : 0)) * num_class);
    thrust::copy(
        params,
        params + (num_col + (fit_intercept ? 1 : 0)) * num_class,
        d_params.begin());

    thrust::device_vector<float> d_preds(num_row);

    ML::GLM::qnDecisionFunction(
        handle,
        d_src_data.data().get(),
        X_col_major,
        int(num_row),
        int(num_col),
        int(num_class),
        fit_intercept,
        d_params.data().get(),
        loss_type,
        d_preds.data().get());

    thrust::copy(
        d_preds.begin(),
        d_preds.end(),
        preds);

    return 0;
}

__host__ int QnDecisionFunctionSparse(
    const float *values,
    const int *indices,
    const int *header,
    size_t num_row,
    size_t num_col,
    size_t num_non_zero,
    size_t num_class,
    bool fit_intercept,
    const float *params,
    int loss_type,
    float *preds)
{
    auto stream_view = cuml4c::stream_allocator::getOrCreateStream();
    raft::handle_t handle;
    cuml4c::handle_utils::initializeHandle(handle, stream_view.value());

    thrust::device_vector<float> d_values(num_non_zero);
    thrust::copy(
        values,
        values + num_non_zero,
        d_values.begin());

    thrust::device_vector<int> d_indices(num_non_zero);
    thrust::copy(
        indices,
        indices + num_non_zero,
        d_indices.begin());

    thrust::device_vector<int> d_header(num_row + 1);
    thrust::copy(
        header,
        header + num_row + 1,
        d_header.begin());

    thrust::device_vector<float> d_params((num_col + (fit_intercept ? 1 : 0)) * num_class);
    thrust::copy(
        params,
        params + (num_col + (fit_intercept ? 1 : 0)) * num_class,
        d_params.begin());

    thrust::device_vector<float> d_preds(num_row);

    ML::GLM::qnDecisionFunctionSparse(
        handle,
        d_values.data().get(),
        d_indices.data().get(),
        d_header.data().get(),
        int(num_non_zero),
        int(num_row),
        int(num_col),
        int(num_class),
        fit_intercept,
        d_params.data().get(),
        loss_type,
        d_preds.data().get());

    thrust::copy(
        d_preds.begin(),
        d_preds.end(),
        preds);

    return 0;
}

__host__ int QnPredict(
    const float *x,
    bool X_col_major,
    size_t num_row,
    size_t num_col,
    size_t num_class,
    bool fit_intercept,
    const float *params,
    int loss_type,
    float *preds)
{
    auto stream_view = cuml4c::stream_allocator::getOrCreateStream();
    raft::handle_t handle;
    cuml4c::handle_utils::initializeHandle(handle, stream_view.value());

    const auto value_length = num_row * num_col;

    thrust::device_vector<float> d_src_data(value_length);
    thrust::copy(
        x,
        x + value_length,
        d_src_data.begin());

    thrust::device_vector<float> d_params((num_col + (fit_intercept ? 1 : 0)) * num_class);
    thrust::copy(
        params,
        params + (num_col + (fit_intercept ? 1 : 0)) * num_class,
        d_params.begin());

    thrust::device_vector<float> d_preds(num_row);

    ML::GLM::qnPredict(
        handle,
        d_src_data.data().get(),
        X_col_major,
        int(num_row),
        int(num_col),
        int(num_class),
        fit_intercept,
        d_params.data().get(),
        loss_type,
        d_preds.data().get());

    thrust::copy(
        d_preds.begin(),
        d_preds.end(),
        preds);

    return 0;
}

__host__ int QnPredictSparse(
    const float *values,
    const int *indices,
    const int *header,
    size_t num_row,
    size_t num_col,
    size_t num_non_zero,
    size_t num_class,
    bool fit_intercept,
    const float *params,
    int loss_type,
    float *preds)
{
    auto stream_view = cuml4c::stream_allocator::getOrCreateStream();
    raft::handle_t handle;
    cuml4c::handle_utils::initializeHandle(handle, stream_view.value());

    thrust::device_vector<float> d_values(num_non_zero);
    thrust::copy(
        values,
        values + num_non_zero,
        d_values.begin());

    thrust::device_vector<int> d_indices(num_non_zero);
    thrust::copy(
        indices,
        indices + num_non_zero,
        d_indices.begin());

    thrust::device_vector<int> d_header(num_row + 1);
    thrust::copy(
        header,
        header + num_row + 1,
        d_header.begin());

    thrust::device_vector<float> d_params((num_col + (fit_intercept ? 1 : 0)) * num_class);
    thrust::copy(
        params,
        params + (num_col + (fit_intercept ? 1 : 0)) * num_class,
        d_params.begin());

    thrust::device_vector<float> d_preds(num_row);

    ML::GLM::qnPredictSparse(
        handle,
        d_values.data().get(),
        d_indices.data().get(),
        d_header.data().get(),
        int(num_non_zero),
        int(num_row),
        int(num_col),
        int(num_class),
        fit_intercept,
        d_params.data().get(),
        loss_type,
        d_preds.data().get());

    thrust::copy(
        d_preds.begin(),
        d_preds.end(),
        preds);

    return 0;
}
