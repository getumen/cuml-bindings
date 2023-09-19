use crate::errors::CumlError;

use super::{
    bindings::{GemmPredict, OlsFit, RidgeFit},
    device_resource::DeviceResource,
};
use anyhow::anyhow;

pub fn ols_fit<'a, 'b>(
    resource: &DeviceResource,
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
            resource.handle,
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
    resource: &DeviceResource,
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
            resource.handle,
            data.as_ptr() as *const f32,
            num_row,
            num_col,
            labels.as_ptr() as *const f32,
            alpha.as_ptr() as *mut f32,
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
    resource: &DeviceResource,
    data: &'a [f32],
    num_row: usize,
    num_col: usize,
    coef: &'b [f32],
    intercept: f32,
) -> Result<Vec<f32>, CumlError> {
    let mut out = vec![0.0; num_row];
    let result = unsafe {
        GemmPredict(
            resource.handle,
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
