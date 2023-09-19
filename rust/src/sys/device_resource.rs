use super::bindings::{CreateDeviceResourceHandle, DeviceResourceHandle, FreeDeviceResourceHandle};

#[derive(Debug)]

pub struct DeviceResource {
    pub(crate) handle: DeviceResourceHandle,
}

impl DeviceResource {
    pub fn new() -> Self {
        let mut handle: DeviceResourceHandle = std::ptr::null_mut();
        unsafe { CreateDeviceResourceHandle(&mut handle) };
        Self { handle }
    }
}

impl Drop for DeviceResource {
    fn drop(&mut self) {
        unsafe { FreeDeviceResourceHandle(self.handle) };
    }
}
