use std::{ffi::CString, path::Path, ptr::null_mut};

use anyhow::{anyhow, Context};

use crate::errors::CumlError;

use super::{
    bindings::{
        DeviceVectorHandleFloat, FILFreeModel, FILGetNumClasses, FILLoadModel, FILModelHandle,
        FILPredict,
    },
    device_vector::DeviceVectorFloat,
};

pub fn fil_load_model<P: AsRef<Path>>(
    model_type: i32,
    model_path: P,
    algo: i32,
    classification: bool,
    threshold: f32,
    storage_type: i32,
    block_per_sm: i32,
    thread_per_tree: i32,
    n_items: i32,
) -> Result<FILModelHandle, CumlError> {
    let model_path = CString::new(model_path.as_ref().to_string_lossy().to_owned().to_string())
        .with_context(|| "get fil model path")?;
    let mut out = null_mut();

    let result = unsafe {
        FILLoadModel(
            model_type,
            model_path.as_ptr(),
            algo,
            classification,
            threshold,
            storage_type,
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

pub fn fil_predict(
    model: FILModelHandle,
    data: &DeviceVectorFloat,
    num_row: usize,
    output_class_probabilities: bool,
) -> Result<DeviceVectorFloat, CumlError> {
    let mut d_preds = DeviceVectorFloat::empty();

    let result = unsafe {
        FILPredict(
            model,
            data.as_ptr() as DeviceVectorHandleFloat,
            num_row,
            output_class_probabilities,
            d_preds.as_mut_ptr() as *mut DeviceVectorHandleFloat,
        )
    };
    if result != 0 {
        Err(anyhow!("fail to predict"))?
    }

    Ok(d_preds)
}

pub fn fil_get_num_class(model: FILModelHandle) -> Result<usize, CumlError> {
    let mut out = 0usize;

    let result = unsafe { FILGetNumClasses(model, &mut out) };

    if result != 0 {
        Err(anyhow!("fail to get num class"))?
    }

    Ok(out)
}
