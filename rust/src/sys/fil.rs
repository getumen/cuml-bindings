use std::{ffi::CString, path::Path, ptr::null_mut};

use anyhow::{anyhow, Context};

use crate::{
    errors::CumlError,
    fil::{Algo, ModelType, StorageType},
};

use super::bindings::{FILFreeModel, FILGetNumClasses, FILLoadModel, FILModelHandle, FILPredict};

pub fn fil_load_model<P: AsRef<Path>>(
    model_type: ModelType,
    model_path: P,
    algo: Algo,
    classification: bool,
    threshold: f32,
    storage_type: StorageType,
    block_per_sm: i32,
    thread_per_tree: i32,
    n_items: i32,
) -> Result<FILModelHandle, CumlError> {
    let model_path = CString::new(model_path.as_ref().to_string_lossy().to_owned().to_string())
        .with_context(|| "get fil model path")?;
    let mut out = null_mut();

    let result = unsafe {
        FILLoadModel(
            model_type as i32,
            model_path.as_ptr(),
            algo as i32,
            classification,
            threshold,
            storage_type as i32,
            block_per_sm,
            thread_per_tree,
            n_items,
            &mut out,
        )
    };
    if result != 0 {
        Err(anyhow!("fail to load model"))?
    }
    Ok(out)
}

pub fn fil_free_model(model: FILModelHandle) -> Result<(), CumlError> {
    let result = unsafe { FILFreeModel(model) };
    if result != 0 {
        Err(anyhow!("fail to free model"))?
    }
    Ok(())
}

pub fn fil_predict<'a>(
    model: FILModelHandle,
    data: &'a [f32],
    num_row: usize,
    output_class_probabilities: bool,
    num_class: usize,
) -> Result<Vec<f32>, CumlError> {
    let mut out = vec![0f32; num_row * num_class];

    let result = unsafe {
        FILPredict(
            model,
            data.as_ptr() as *const f32,
            num_row,
            output_class_probabilities,
            out.as_mut_ptr() as *mut f32,
        )
    };
    if result != 0 {
        Err(anyhow!("fail to predict"))?
    }

    Ok(out)
}

pub fn fil_get_num_class(model: FILModelHandle) -> Result<usize, CumlError> {
    let mut out = 0usize;

    let result = unsafe { FILGetNumClasses(model, &mut out) };

    if result != 0 {
        Err(anyhow!("fail to get num class"))?
    }

    Ok(out)
}
