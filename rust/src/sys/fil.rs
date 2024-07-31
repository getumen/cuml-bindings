use std::{ffi::CString, path::Path, ptr::null_mut};

use anyhow::{anyhow, Context};

use crate::errors::CumlError;

use super::{
    bindings::{FILFreeModel, FILLoadModel, FILModelHandle, FILPredict},
    device_resource::DeviceResource,
};

pub fn fil_load_model<P: AsRef<Path>>(
    resource: &DeviceResource,
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
            resource.handle,
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

pub fn fil_free_model(resource: &DeviceResource, model: FILModelHandle) -> Result<(), CumlError> {
    let result = unsafe { FILFreeModel(resource.handle, model) };
    if result != 0 {
        Err(anyhow!("fail to free model"))?
    }
    Ok(())
}

pub fn fil_predict(
    resource: &DeviceResource,
    model: FILModelHandle,
    data: &[f32],
    num_row: usize,
    output_class_probabilities: bool,
    preds: &mut [f32],
) -> Result<(), CumlError> {
    let result = unsafe {
        FILPredict(
            resource.handle,
            model,
            data.as_ptr() as *const f32,
            num_row,
            output_class_probabilities,
            preds.as_mut_ptr() as *mut f32,
        )
    };
    if result != 0 {
        Err(anyhow!("fail to predict"))?
    }

    Ok(())
}
