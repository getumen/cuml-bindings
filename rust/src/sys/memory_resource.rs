use crate::errors::CumlError;
use anyhow::anyhow;

use std::ptr::null_mut;

use super::bindings::{
    DeviceMemoryResource, ResetMemoryResource, UseArenaMemoryResource, UseBinningMemoryResource,
    UsePoolMemoryResource,
};

pub struct MemoryResource {
    resource: DeviceMemoryResource,
    resource_type: i32,
}

impl MemoryResource {
    pub fn use_pool_memory_resource(
        initial_pool_size: usize,
        maximum_pool_size: usize,
    ) -> Result<Self, CumlError> {
        let mut resource: DeviceMemoryResource = null_mut();
        let ret =
            unsafe { UsePoolMemoryResource(initial_pool_size, maximum_pool_size, &mut resource) };
        if ret != 0 {
            Err(anyhow!("fail to use pool memory resource"))?
        }
        Ok(Self {
            resource,
            resource_type: 0,
        })
    }

    pub fn use_binning_memory_resource(
        min_size_exponent: i8,
        max_size_exponent: i8,
    ) -> Result<Self, CumlError> {
        let mut resource: DeviceMemoryResource = null_mut();
        let ret = unsafe {
            UseBinningMemoryResource(min_size_exponent, max_size_exponent, &mut resource)
        };
        if ret != 0 {
            Err(anyhow!("fail to use binning memory resource"))?
        }
        Ok(Self {
            resource,
            resource_type: 1,
        })
    }

    pub fn use_arena_memoryg_resource() -> Result<Self, CumlError> {
        let mut resource: DeviceMemoryResource = null_mut();
        let ret = unsafe { UseArenaMemoryResource(&mut resource) };
        if ret != 0 {
            Err(anyhow!("fail to use arena memory resource"))?
        }
        Ok(Self {
            resource,
            resource_type: 2,
        })
    }
}

impl Drop for MemoryResource {
    fn drop(&mut self) {
        unsafe {
            ResetMemoryResource(self.resource, self.resource_type);
        }
    }
}
