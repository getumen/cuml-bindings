use crate::errors::CumlError;

use super::bindings::{
    GemmPredict, OlsFit, QnDecisionFunction, QnDecisionFunctionSparse, QnFit, QnFitSparse,
    QnPredict, QnPredictSparse, RidgeFit,
};
use anyhow::anyhow;

pub fn ols_fit<'a, 'b>(
    data: &'a [f32],
    num_row: usize,
    num_col: usize,
    labels: &'b [f32],
    fit_intercept: bool,
    normalize: bool,
    algo: i32,
) -> Result<(Vec<f32>, f32), CumlError> {
    let mut coef = vec![0.0; num_col];
    let mut intercept = 0.0;
    let result = unsafe {
        OlsFit(
            data.as_ptr() as *const f32,
            num_row,
            num_col,
            labels.as_ptr() as *const f32,
            fit_intercept,
            normalize,
            algo,
            coef.as_mut_ptr() as *mut f32,
            &mut intercept,
        )
    };

    if result != 0 {
        Err(anyhow!("fail to OlsFit"))?
    }

    Ok((coef, intercept))
}

pub fn ridge_fit<'a, 'b>(
    data: &'a [f32],
    num_row: usize,
    num_col: usize,
    labels: &'b [f32],
    alpha: f32,
    fit_intercept: bool,
    normalize: bool,
    algo: i32,
) -> Result<(Vec<f32>, f32), CumlError> {
    let alpha = vec![alpha];
    let mut coef = vec![0.0; num_col];
    let mut intercept = 0.0;
    let result = unsafe {
        RidgeFit(
            data.as_ptr() as *const f32,
            num_row,
            num_col,
            labels.as_ptr() as *const f32,
            alpha.as_ptr() as *const f32,
            alpha.len(),
            fit_intercept,
            normalize,
            algo,
            coef.as_mut_ptr() as *mut f32,
            &mut intercept,
        )
    };

    if result != 0 {
        Err(anyhow!("fail to RidgeFit"))?
    }

    Ok((coef, intercept))
}

pub fn gemm_predict<'a, 'b>(
    data: &'a [f32],
    num_row: usize,
    num_col: usize,
    coef: &'b [f32],
    intercept: f32,
) -> Result<Vec<f32>, CumlError> {
    let mut out = vec![0.0; num_row];
    let result = unsafe {
        GemmPredict(
            data.as_ptr() as *const f32,
            num_row,
            num_col,
            coef.as_ptr() as *const f32,
            intercept,
            out.as_mut_ptr() as *mut f32,
        )
    };

    if result != 0 {
        Err(anyhow!("fail to GemmPredict"))?
    }

    Ok(out)
}

pub fn qn_fit<'a, 'b, 'c>(
    data: &'a [f32],
    num_row: usize,
    num_col: usize,
    x_col_major: bool,
    labels: &'b [f32],
    num_classes: usize,
    sample_weight: Option<&'c [f32]>,
    loss_type: i32,
    fit_intercept: bool,
    l1: f32,
    l2: f32,
    max_iter: i32,
    grad_tol: f32,
    change_tol: f32,
    linesearch_max_iter: i32,
    lbfgs_memory: i32,
    verbosity: i32,
) -> Result<(Vec<f32>, f32, i32), CumlError> {
    let mut coef = vec![0.0; num_col];
    let mut loss = 0.0;
    let mut n_iter = 0;
    let result = unsafe {
        QnFit(
            data.as_ptr() as *const f32,
            num_row,
            num_col,
            x_col_major,
            labels.as_ptr() as *const f32,
            num_classes,
            loss_type,
            sample_weight.map_or(std::ptr::null(), |x| x.as_ptr()),
            fit_intercept,
            l1,
            l2,
            max_iter,
            grad_tol,
            change_tol,
            linesearch_max_iter,
            lbfgs_memory,
            verbosity,
            coef.as_mut_ptr() as *mut f32,
            &mut loss,
            &mut n_iter,
        )
    };

    if result != 0 {
        Err(anyhow!("fail to QnFit"))?
    }

    Ok((coef, loss, n_iter))
}

pub fn qn_fit_sparse<'a, 'b, 'c, 'd>(
    data: &'a [f32],
    indices: &'b [i32],
    header: &'c [i32],
    num_row: usize,
    num_col: usize,
    num_non_zero: usize,
    labels: &'b [f32],
    num_classes: usize,
    sample_weight: Option<&'d [f32]>,
    loss_type: i32,
    fit_intercept: bool,
    l1: f32,
    l2: f32,
    max_iter: i32,
    grad_tol: f32,
    change_tol: f32,
    linesearch_max_iter: i32,
    lbfgs_memory: i32,
    verbosity: i32,
) -> Result<(Vec<f32>, f32, i32), CumlError> {
    let mut coef = vec![0.0; (num_col + if fit_intercept { 1 } else { 0 }) * num_classes];
    let mut loss = 0.0;
    let mut n_iter = 0;
    let result = unsafe {
        QnFitSparse(
            data.as_ptr() as *const f32,
            indices.as_ptr() as *const i32,
            header.as_ptr() as *const i32,
            num_row,
            num_col,
            num_non_zero,
            labels.as_ptr() as *const f32,
            num_classes,
            loss_type,
            sample_weight.map_or(std::ptr::null(), |x| x.as_ptr()),
            fit_intercept,
            l1,
            l2,
            max_iter,
            grad_tol,
            change_tol,
            linesearch_max_iter,
            lbfgs_memory,
            verbosity,
            coef.as_mut_ptr() as *mut f32,
            &mut loss,
            &mut n_iter,
        )
    };

    if result != 0 {
        Err(anyhow!("fail to QnFitSparse"))?
    }

    Ok((coef, loss, n_iter))
}

pub fn qn_decision_function<'a, 'b>(
    data: &'a [f32],
    num_row: usize,
    num_col: usize,
    x_col_major: bool,
    num_class: usize,
    fit_intercept: bool,
    coef: &'b [f32],
    loss_type: i32,
) -> Result<Vec<f32>, CumlError> {
    let mut out = vec![0.0; num_row];
    let result = unsafe {
        QnDecisionFunction(
            data.as_ptr() as *const f32,
            x_col_major,
            num_row,
            num_col,
            num_class,
            fit_intercept,
            coef.as_ptr() as *const f32,
            loss_type,
            out.as_mut_ptr() as *mut f32,
        )
    };

    if result != 0 {
        Err(anyhow!("fail to QnDecisionFunction"))?
    }

    Ok(out)
}

pub fn qn_decision_function_sparse<'a, 'b, 'c, 'd>(
    data: &'a [f32],
    indices: &'b [i32],
    header: &'c [i32],
    num_row: usize,
    num_col: usize,
    num_non_zero: usize,
    num_class: usize,
    fit_intercept: bool,
    coef: &'d [f32],
    loss_type: i32,
) -> Result<Vec<f32>, CumlError> {
    let mut out = vec![0.0; num_row];
    let result = unsafe {
        QnDecisionFunctionSparse(
            data.as_ptr() as *const f32,
            indices.as_ptr() as *const i32,
            header.as_ptr() as *const i32,
            num_row,
            num_col,
            num_non_zero,
            num_class,
            fit_intercept,
            coef.as_ptr() as *const f32,
            loss_type,
            out.as_mut_ptr() as *mut f32,
        )
    };

    if result != 0 {
        Err(anyhow!("fail to QnDecisionFunctionSparse"))?
    }

    Ok(out)
}

pub fn qn_predict<'a, 'b>(
    data: &'a [f32],
    num_row: usize,
    num_col: usize,
    x_col_major: bool,
    num_class: usize,
    fit_intercept: bool,
    coef: &'b [f32],
    loss_type: i32,
) -> Result<Vec<f32>, CumlError> {
    let mut out = vec![0.0; num_row];
    let result = unsafe {
        QnPredict(
            data.as_ptr() as *const f32,
            x_col_major,
            num_row,
            num_col,
            num_class,
            fit_intercept,
            coef.as_ptr() as *const f32,
            loss_type,
            out.as_mut_ptr() as *mut f32,
        )
    };

    if result != 0 {
        Err(anyhow!("fail to QnDecisionFunction"))?
    }

    Ok(out)
}

pub fn qn_predict_sparse<'a, 'b, 'c, 'd>(
    data: &'a [f32],
    indices: &'b [i32],
    header: &'c [i32],
    num_row: usize,
    num_col: usize,
    num_non_zero: usize,
    num_class: usize,
    fit_intercept: bool,
    coef: &'d [f32],
    loss_type: i32,
) -> Result<Vec<f32>, CumlError> {
    let mut out = vec![0.0; num_row];
    let result = unsafe {
        QnPredictSparse(
            data.as_ptr() as *const f32,
            indices.as_ptr() as *const i32,
            header.as_ptr() as *const i32,
            num_row,
            num_col,
            num_non_zero,
            num_class,
            fit_intercept,
            coef.as_ptr() as *const f32,
            loss_type,
            out.as_mut_ptr() as *mut f32,
        )
    };

    if result != 0 {
        Err(anyhow!("fail to QnDecisionFunctionSparse"))?
    }

    Ok(out)
}
