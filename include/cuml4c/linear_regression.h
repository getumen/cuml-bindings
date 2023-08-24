#ifdef __cplusplus
#define EXTERN_C extern "C"
#include <cstddef>
#else
#define EXTERN_C
#include <stdbool.h>
#include <stdio.h>
#endif

EXTERN_C int OlsFit(
    const float *x,
    size_t num_row,
    size_t num_col,
    const float *labels,
    bool fit_intercept,
    bool normalize,
    int algo,
    float *coef,
    float *intercept);

EXTERN_C int RidgeFit(
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
    float *intercept);

EXTERN_C int GemmPredict(
    const float *x,
    size_t num_row,
    size_t num_col,
    const float *coef,
    float intercept,
    float *preds);

EXTERN_C int QnFit(
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
    int *num_iters);

EXTERN_C int QnFitSparse(
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
    int *num_iters);

EXTERN_C int QnDecisionFunction(
    const float *x,
    bool X_col_major,
    size_t num_row,
    size_t num_col,
    size_t num_class,
    bool fit_intercept,
    const float *params,
    int loss_type,
    float *preds);

EXTERN_C int QnDecisionFunctionSparse(
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
    float *preds);

EXTERN_C int QnPredict(
    const float *x,
    bool X_col_major,
    size_t num_row,
    size_t num_col,
    size_t num_class,
    bool fit_intercept,
    const float *params,
    int loss_type,
    float *preds);

EXTERN_C int QnPredictSparse(
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
    float *preds);
