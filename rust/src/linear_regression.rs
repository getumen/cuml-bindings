use crate::{errors::CumlError, sys::linear_regression};

use anyhow::anyhow;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum Algo {
    Svd = 0,
    Eig = 1,
    Qr = 2,
}

#[derive(Debug, Clone)]
pub struct LinearRegression {
    coef: Option<Vec<f32>>,
    intercept: Option<f32>,
    fit_intercept: bool,
    normalize: bool,
    algo: Algo,
}

impl LinearRegression {
    pub fn new(fit_intercept: bool, normalize: bool, algo: Algo) -> Self {
        Self {
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
        linear_regression::gemm_predict(data, num_row, num_col, coef, intercept)
    }

    pub fn get_params(&self) -> (&[f32], f32) {
        (self.coef.as_ref().unwrap(), self.intercept.unwrap_or(0.0))
    }

    pub fn set_params(&mut self, coef: &[f32], intercept: f32) {
        self.coef = Some(coef.to_vec());
        self.intercept = Some(intercept);
    }
}

#[derive(Debug, Clone)]
pub struct Ridge {
    coef: Option<Vec<f32>>,
    intercept: Option<f32>,
    alpha: f32,
    fit_intercept: bool,
    normalize: bool,
    algo: Algo,
}

impl Ridge {
    pub fn new(alpha: f32, fit_intercept: bool, normalize: bool, algo: Algo) -> Self {
        Self {
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
        linear_regression::gemm_predict(data, num_row, num_col, coef, intercept)
    }

    pub fn get_params(&self) -> (&[f32], f32) {
        (self.coef.as_ref().unwrap(), self.intercept.unwrap_or(0.0))
    }

    pub fn set_params(&mut self, coef: &[f32], intercept: f32) {
        self.coef = Some(coef.to_vec());
        self.intercept = Some(intercept);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]

pub enum LossType {
    Logistic = 0,
    Normal = 1,
    Multinomiral = 2,
}

#[derive(Debug, Clone)]
pub struct QnGlm {
    coef: Option<Vec<f32>>,
    num_class: Option<usize>,
    loss_type: LossType,
    fit_intercept: bool,
    l1: f32,
    l2: f32,
    verbosity: i32,
}

impl QnGlm {
    pub fn new(loss_type: LossType, fit_intercept: bool, l1: f32, l2: f32, verbosity: i32) -> Self {
        Self {
            coef: None,
            num_class: None,
            fit_intercept,
            loss_type,
            l1,
            l2,
            verbosity,
        }
    }

    pub fn fit<'a, 'b, 'c>(
        &mut self,
        data: &'a [f32],
        num_row: usize,
        num_col: usize,
        x_col_major: bool,
        labels: &'b [f32],
        num_classes: usize,
        sample_weight: Option<&'c [f32]>,
        max_iter: i32,
        grad_tol: f32,
        change_tol: f32,
        linesearch_max_iter: i32,
        lbfgs_memory: i32,
    ) -> Result<f32, CumlError> {
        let (coef, loss, _) = linear_regression::qn_fit(
            data,
            num_row,
            num_col,
            x_col_major,
            labels,
            num_classes,
            sample_weight,
            self.loss_type as i32,
            self.fit_intercept,
            self.l1,
            self.l2,
            max_iter,
            grad_tol,
            change_tol,
            linesearch_max_iter,
            lbfgs_memory,
            self.verbosity,
        )?;

        self.coef = Some(coef);
        self.num_class = Some(num_classes);
        Ok(loss)
    }

    pub fn fit_sparse<'a, 'b, 'c, 'd>(
        &mut self,
        data: &'a [f32],
        indices: &'b [i32],
        header: &'c [i32],
        num_row: usize,
        num_col: usize,
        num_non_zero: usize,
        labels: &'b [f32],
        num_classes: usize,
        sample_weight: Option<&'d [f32]>,
        max_iter: i32,
        grad_tol: f32,
        change_tol: f32,
        linesearch_max_iter: i32,
        lbfgs_memory: i32,
    ) -> Result<f32, CumlError> {
        let (coef, loss, _) = linear_regression::qn_fit_sparse(
            data,
            indices,
            header,
            num_row,
            num_col,
            num_non_zero,
            labels,
            num_classes,
            sample_weight,
            self.loss_type as i32,
            self.fit_intercept,
            self.l1,
            self.l2,
            max_iter,
            grad_tol,
            change_tol,
            linesearch_max_iter,
            lbfgs_memory,
            self.verbosity,
        )?;
        self.coef = Some(coef);
        self.num_class = Some(num_classes);

        Ok(loss)
    }

    pub fn decision_function<'a>(
        &self,
        data: &'a [f32],
        num_row: usize,
        num_col: usize,
        x_col_major: bool,
    ) -> Result<Vec<f32>, CumlError> {
        let out = linear_regression::qn_decision_function(
            data,
            num_row,
            num_col,
            x_col_major,
            *self
                .num_class
                .as_ref()
                .ok_or_else(|| anyhow!("num_class is None"))?,
            self.fit_intercept,
            self.coef.as_ref().ok_or_else(|| anyhow!("coef is None"))?,
            self.loss_type as i32,
        )?;

        Ok(out)
    }

    pub fn decision_function_sparse<'a, 'b, 'c>(
        &self,
        data: &'a [f32],
        indices: &'b [i32],
        header: &'c [i32],
        num_row: usize,
        num_col: usize,
        num_non_zero: usize,
    ) -> Result<Vec<f32>, CumlError> {
        let out = linear_regression::qn_decision_function_sparse(
            data,
            indices,
            header,
            num_row,
            num_col,
            num_non_zero,
            *self
                .num_class
                .as_ref()
                .ok_or_else(|| anyhow!("num_class is None"))?,
            self.fit_intercept,
            self.coef.as_ref().ok_or_else(|| anyhow!("coef is None"))?,
            self.loss_type as i32,
        )?;

        Ok(out)
    }

    pub fn predict<'a, 'b>(
        &self,
        data: &'a [f32],
        num_row: usize,
        num_col: usize,
        x_col_major: bool,
    ) -> Result<Vec<f32>, CumlError> {
        let out = linear_regression::qn_predict(
            data,
            num_row,
            num_col,
            x_col_major,
            *self
                .num_class
                .as_ref()
                .ok_or_else(|| anyhow!("num_class is None"))?,
            self.fit_intercept,
            self.coef.as_ref().ok_or_else(|| anyhow!("coef is None"))?,
            self.loss_type as i32,
        )?;

        Ok(out)
    }

    pub fn predict_sparse<'a, 'b, 'c, 'd>(
        &self,
        data: &'a [f32],
        indices: &'b [i32],
        header: &'c [i32],
        num_row: usize,
        num_col: usize,
        num_non_zero: usize,
    ) -> Result<Vec<f32>, CumlError> {
        let out = linear_regression::qn_predict_sparse(
            data,
            indices,
            header,
            num_row,
            num_col,
            num_non_zero,
            *self
                .num_class
                .as_ref()
                .ok_or_else(|| anyhow!("num_class is None"))?,
            self.fit_intercept,
            self.coef.as_ref().ok_or_else(|| anyhow!("coef is None"))?,
            self.loss_type as i32,
        )?;

        Ok(out)
    }

    pub fn get_params(&self) -> Option<&Vec<f32>> {
        self.coef.as_ref()
    }

    pub fn set_params(&mut self, coef: Vec<f32>) {
        self.coef = Some(coef);
    }
}
