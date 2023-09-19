use crate::{
    errors::CumlError,
    sys::{device_resource::DeviceResource, linear_regression},
};

use anyhow::anyhow;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum Algo {
    Svd = 0,
    Eig = 1,
    Qr = 2,
}

#[derive(Debug)]
pub struct LinearRegression {
    device_resource: DeviceResource,
    coef: Option<Vec<f32>>,
    intercept: Option<f32>,
    fit_intercept: bool,
    normalize: bool,
    algo: Algo,
}

impl LinearRegression {
    pub fn new(fit_intercept: bool, normalize: bool, algo: Algo) -> Self {
        let device_resource = DeviceResource::new();
        Self {
            device_resource,
            coef: None,
            intercept: None,
            fit_intercept,
            normalize,
            algo,
        }
    }

    pub fn fit<'a, 'b>(
        &mut self,
        data: &'a [f32],
        num_row: usize,
        num_col: usize,
        labels: &'b [f32],
    ) -> Result<(), CumlError> {
        let (coef, intercept) = linear_regression::ols_fit(
            &self.device_resource,
            data,
            num_row,
            num_col,
            labels,
            self.fit_intercept,
            self.normalize,
            self.algo as i32,
        )?;

        self.coef = Some(coef);
        self.intercept = Some(intercept);
        Ok(())
    }

    pub fn predict<'a>(
        &self,
        data: &'a [f32],
        num_row: usize,
        num_col: usize,
    ) -> Result<Vec<f32>, CumlError> {
        let coef = self.coef.as_ref().ok_or_else(|| anyhow!("coef is None"))?;
        let intercept = self.intercept.ok_or_else(|| anyhow!("intercept is None"))?;
        linear_regression::gemm_predict(
            &self.device_resource,
            data,
            num_row,
            num_col,
            coef,
            intercept,
        )
    }

    pub fn get_params(&self) -> (&[f32], f32) {
        (self.coef.as_ref().unwrap(), self.intercept.unwrap_or(0.0))
    }

    pub fn set_params(&mut self, coef: &[f32], intercept: f32) {
        self.coef = Some(coef.to_vec());
        self.intercept = Some(intercept);
    }
}

#[derive(Debug)]
pub struct Ridge {
    device_resource: DeviceResource,
    coef: Option<Vec<f32>>,
    intercept: Option<f32>,
    alpha: f32,
    fit_intercept: bool,
    normalize: bool,
    algo: Algo,
}

impl Ridge {
    pub fn new(alpha: f32, fit_intercept: bool, normalize: bool, algo: Algo) -> Self {
        let device_resource = DeviceResource::new();
        Self {
            device_resource,
            coef: None,
            intercept: None,
            alpha,
            fit_intercept,
            normalize,
            algo,
        }
    }

    pub fn fit<'a, 'b>(
        &mut self,
        data: &'a [f32],
        num_row: usize,
        num_col: usize,
        labels: &'b [f32],
    ) -> Result<(), CumlError> {
        let (coef, intercept) = linear_regression::ridge_fit(
            &self.device_resource,
            data,
            num_row,
            num_col,
            labels,
            self.alpha,
            self.fit_intercept,
            self.normalize,
            self.algo as i32,
        )?;

        self.coef = Some(coef);
        self.intercept = Some(intercept);
        Ok(())
    }

    pub fn predict<'a>(
        &self,
        data: &'a [f32],
        num_row: usize,
        num_col: usize,
    ) -> Result<Vec<f32>, CumlError> {
        let coef = self.coef.as_ref().ok_or_else(|| anyhow!("coef is None"))?;
        let intercept = self.intercept.ok_or_else(|| anyhow!("intercept is None"))?;
        linear_regression::gemm_predict(
            &self.device_resource,
            data,
            num_row,
            num_col,
            coef,
            intercept,
        )
    }

    pub fn get_params(&self) -> (&[f32], f32) {
        (self.coef.as_ref().unwrap(), self.intercept.unwrap_or(0.0))
    }

    pub fn set_params(&mut self, coef: &[f32], intercept: f32) {
        self.coef = Some(coef.to_vec());
        self.intercept = Some(intercept);
    }
}
