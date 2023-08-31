package rawcuml4go

// #cgo LDFLAGS: -lcuml4c -lcuml++ -lcuml -lcumlprims
// #include <stdlib.h>
// #include "cuml4c/device_vector.h"
import "C"
import "errors"

var (
	ErrRawDeviceVector   = errors.New("raw api: fail to copy device to host")
	ErrRawHostVector     = errors.New("raw api: fail to copy host to device")
	ErrCloseDeviceVector = errors.New("raw api: fail to close device vector")
	ErrGetSize           = errors.New("raw api: fail to get size")
)

type DeviceVectorFloat struct {
	pointer C.DeviceVectorHandleFloat
}
type DeviceVectorInt struct {
	pointer C.DeviceVectorHandleInt
}

func NewDeviceVectorFloatEmpty() *DeviceVectorFloat {
	var pointer C.DeviceVectorHandleFloat
	return &DeviceVectorFloat{
		pointer: pointer,
	}
}

func NewDeviceVectorFloat(
	data []float32,
) (*DeviceVectorFloat, error) {

	var pointer C.DeviceVectorHandleFloat
	ret := C.HostVectorToDeviceVectorFloat(
		(*C.float)(&data[0]),
		(C.ulong)(len(data)),
		&pointer,
	)

	if ret != 0 {
		return nil, ErrRawHostVector
	}

	return &DeviceVectorFloat{
		pointer: pointer,
	}, nil
}

func (d *DeviceVectorFloat) Close() error {
	ret := C.DeviceVectorFloatFree(
		d.pointer,
	)

	if ret != 0 {
		return ErrCloseDeviceVector
	}

	return nil
}

func (d *DeviceVectorFloat) GetSize() (int, error) {
	var size uint64
	ret := C.DeviceVectorFloatGetSize(
		d.pointer,
		(*C.ulong)(&size),
	)

	if ret != 0 {
		return 0, ErrGetSize
	}

	return int(size), nil
}

func (d *DeviceVectorFloat) ToHost() ([]float32, error) {
	size, err := d.GetSize()
	if err != nil {
		return nil, err
	}

	result := make([]float32, size)

	ret := C.DeviceVectorToHostVectorFloat(
		d.pointer,
		(*C.float)(&result[0]),
	)

	if ret != 0 {
		return nil, ErrRawDeviceVector
	}
	return result, nil
}

func (d *DeviceVectorFloat) ToHostInPlace(out []float32) error {
	ret := C.DeviceVectorToHostVectorFloat(
		d.pointer,
		(*C.float)(&out[0]),
	)

	if ret != 0 {
		return ErrRawDeviceVector
	}
	return nil
}

func NewDeviceVectorIntEmpty() *DeviceVectorInt {
	var pointer C.DeviceVectorHandleInt
	return &DeviceVectorInt{
		pointer: pointer,
	}
}

func NewDeviceVectorInt(
	data []int32,
) (*DeviceVectorInt, error) {

	var pointer C.DeviceVectorHandleInt
	ret := C.HostVectorToDeviceVectorInt(
		(*C.int)(&data[0]),
		(C.ulong)(len(data)),
		&pointer,
	)

	if ret != 0 {
		return nil, ErrRawHostVector
	}

	return &DeviceVectorInt{
		pointer: pointer,
	}, nil
}

func (d *DeviceVectorInt) Close() error {
	ret := C.DeviceVectorIntFree(
		d.pointer,
	)

	if ret != 0 {
		return ErrCloseDeviceVector
	}

	return nil
}

func (d *DeviceVectorInt) GetSize() (int, error) {
	var size uint64
	ret := C.DeviceVectorIntGetSize(
		d.pointer,
		(*C.ulong)(&size),
	)

	if ret != 0 {
		return 0, ErrGetSize
	}

	return int(size), nil
}

func (d *DeviceVectorInt) ToHost() ([]int32, error) {
	size, err := d.GetSize()
	if err != nil {
		return nil, err
	}

	result := make([]int32, size)

	ret := C.DeviceVectorToHostVectorInt(
		d.pointer,
		(*C.int)(&result[0]),
	)

	if ret != 0 {
		return nil, ErrRawDeviceVector
	}
	return result, nil
}

func (d *DeviceVectorInt) ToHostInPlace(out []int32) error {
	ret := C.DeviceVectorToHostVectorInt(
		d.pointer,
		(*C.int)(&out[0]),
	)

	if ret != 0 {
		return ErrRawDeviceVector
	}
	return nil
}
