use std::{os::raw::c_void, ptr::null_mut};

use crate::errors::CumlError;
use anyhow::anyhow;

use super::bindings::{
    DeviceVectorFloatCreate, DeviceVectorFloatFree, DeviceVectorFloatGetSize,
    DeviceVectorHandleFloat, DeviceVectorHandleInt, DeviceVectorIntCreate,
    DeviceVectorToHostVectorFloat, DeviceVectorToHostVectorInt, HostVectorToDeviceVectorFloat,
    HostVectorToDeviceVectorInt,
};

pub struct DeviceVectorFloat {
    handle: DeviceVectorHandleFloat,
}

impl DeviceVectorFloat {
    pub fn new(size: usize) -> Result<Self, CumlError> {
        let mut handle = null_mut();
        let result: i32 = unsafe { DeviceVectorFloatCreate(size, &mut handle) };
        if result != 0 {
            Err(anyhow!("fail to create device_vector<float>"))?
        }
        Ok(Self { handle })
    }

    pub fn from_slice(data: &[f32]) -> Result<Self, CumlError> {
        let mut handle = null_mut();
        let result: i32 = unsafe {
            HostVectorToDeviceVectorFloat(data.as_ptr() as *const f32, data.len(), &mut handle)
        };

        if result != 0 {
            Err(anyhow!("fail to create device_vector<float>"))?
        }

        Ok(Self { handle })
    }

    pub fn get_size(&self) -> Result<usize, CumlError> {
        let mut size = 0usize;
        let result: i32 = unsafe { DeviceVectorFloatGetSize(self.handle, &mut size) };

        if result != 0 {
            Err(anyhow!("fail to get device_vector<float> size"))?
        }

        Ok(size)
    }

    pub fn to_host(&self) -> Result<Vec<f32>, CumlError> {
        let mut data = vec![0f32; self.get_size()?];
        let result: i32 =
            unsafe { DeviceVectorToHostVectorFloat(self.handle, data.as_mut_ptr() as *mut f32) };

        if result != 0 {
            Err(anyhow!("fail to copy device_vector<float> to host"))?
        }

        Ok(data)
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.handle as *const c_void
    }

    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.handle as *mut c_void
    }
}

impl Drop for DeviceVectorFloat {
    fn drop(&mut self) {
        unsafe {
            DeviceVectorFloatFree(self.handle);
        }
    }
}

pub struct DeviceVectorInt {
    handle: DeviceVectorHandleInt,
}

impl DeviceVectorInt {
    pub fn new(size: usize) -> Result<Self, CumlError> {
        let mut handle = null_mut();
        let result: i32 = unsafe { DeviceVectorIntCreate(size, &mut handle) };
        if result != 0 {
            Err(anyhow!("fail to create device_vector<int>"))?
        }
        Ok(Self { handle })
    }

    pub fn from_slice(data: &[i32]) -> Result<Self, CumlError> {
        let mut handle = null_mut();
        let result: i32 = unsafe {
            HostVectorToDeviceVectorInt(data.as_ptr() as *const i32, data.len(), &mut handle)
        };

        if result != 0 {
            Err(anyhow!("fail to create device_vector<int>"))?
        }

        Ok(Self { handle })
    }

    pub fn get_size(&self) -> Result<usize, CumlError> {
        let mut size = 0usize;
        let result: i32 = unsafe { DeviceVectorFloatGetSize(self.handle, &mut size) };

        if result != 0 {
            Err(anyhow!("fail to get device_vector<int> size"))?
        }

        Ok(size)
    }

    pub fn to_host(&self) -> Result<Vec<i32>, CumlError> {
        let mut data = vec![0i32; self.get_size()?];
        let result =
            unsafe { DeviceVectorToHostVectorInt(self.handle, data.as_mut_ptr() as *mut i32) };

        if result != 0 {
            Err(anyhow!("fail to copy device_vector<int> to host"))?
        }

        Ok(data)
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.handle as *const c_void
    }

    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.handle as *mut c_void
    }
}

impl Drop for DeviceVectorInt {
    fn drop(&mut self) {
        unsafe {
            DeviceVectorFloatFree(self.handle);
        }
    }
}
