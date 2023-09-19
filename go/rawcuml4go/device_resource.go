package rawcuml4go

// #cgo LDFLAGS: -lcuml4c -lcuml++ -lcuml
// #include <stdlib.h>
// #include "cuml4c/device_resource_handle.h"
import "C"
import "errors"

var (
	ErrCreateDeviceResource = errors.New("raw api: fail to create device resource")
	ErrCloseDeviceResource  = errors.New("raw api: fail to close device resource")
)

type DeviceResource struct {
	pointer C.DeviceResourceHandle
}

func NewDeviceResource() (*DeviceResource, error) {
	var pointer C.DeviceResourceHandle
	ret := C.CreateDeviceResourceHandle(&pointer)
	if ret != 0 {
		return nil, ErrCreateDeviceResource
	}
	return &DeviceResource{
		pointer: pointer,
	}, nil
}

func (d *DeviceResource) Close() error {
	ret := C.FreeDeviceResourceHandle(d.pointer)
	if ret != 0 {
		return ErrCloseDeviceResource
	}
	return nil
}
