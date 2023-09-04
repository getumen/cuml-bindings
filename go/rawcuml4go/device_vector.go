package rawcuml4go

// #cgo LDFLAGS: -lcuml4c -lcuml++ -lcuml -lcumlprims
// #include <stdlib.h>
// #include "cuml4c/device_vector.h"
import "C"
import "errors"

var (
	ErrRawCreateDeviceVector = errors.New("raw api: fail to create device vector")
	ErrRawDeviceVector       = errors.New("raw api: fail to copy device to host")
	ErrRawHostVector         = errors.New("raw api: fail to copy host to device")
	ErrRawCloseDeviceVector  = errors.New("raw api: fail to close device vector")
	ErrRawGetSize            = errors.New("raw api: fail to get size")
)

type DeviceVectorFloat struct {
	pointer C.DeviceVectorHandleFloat
}
type DeviceVectorInt struct {
	pointer C.DeviceVectorHandleInt
}

func NewDeviceVectorFloat(size int) (*DeviceVectorFloat, error) {
	var pointer C.DeviceVectorHandleFloat
	ret := C.DeviceVectorFloatCreate(
		C.ulong(size),
		&pointer,
	)
	if ret != 0 {
		return nil, ErrRawCreateDeviceVector
	}
	return &DeviceVectorFloat{
		pointer: pointer,
	}, nil
}

func HostToDeviceFloat(
	data []float32,
	vector *DeviceVectorFloat,
) error {
	ret := C.HostVectorToDeviceVectorFloat(
		(*C.float)(&data[0]),
		(C.ulong)(len(data)),
		&vector.pointer,
	)

	if ret != 0 {
		return ErrRawHostVector
	}

	return nil
}

func NewDeviceVectorFloatFromData(
	data []float32,
) (*DeviceVectorFloat, error) {

	vector, err := NewDeviceVectorFloat(len(data))
	if err != nil {
		return nil, err
	}
	err = HostToDeviceFloat(
		data,
		vector,
	)
	if err != nil {
		return nil, err
	}

	return vector, nil
}

func (d *DeviceVectorFloat) Close() error {
	ret := C.DeviceVectorFloatFree(
		d.pointer,
	)

	if ret != 0 {
		return ErrRawCloseDeviceVector
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
		return 0, ErrRawGetSize
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

func NewDeviceVectorInt(size int) (*DeviceVectorInt, error) {
	var pointer C.DeviceVectorHandleInt
	ret := C.DeviceVectorIntCreate(
		C.ulong(size),
		&pointer,
	)
	if ret != 0 {
		return nil, ErrRawCreateDeviceVector
	}
	return &DeviceVectorInt{
		pointer: pointer,
	}, nil
}

func HostToDeviceInt(
	data []int32,
	vector *DeviceVectorInt,
) error {
	ret := C.HostVectorToDeviceVectorInt(
		(*C.int)(&data[0]),
		(C.ulong)(len(data)),
		&vector.pointer,
	)

	if ret != 0 {
		return ErrRawHostVector
	}

	return nil
}

func NewDeviceVectorIntFromData(
	data []int32,
) (*DeviceVectorInt, error) {

	vector, err := NewDeviceVectorInt(len(data))
	if err != nil {
		return nil, err
	}
	err = HostToDeviceInt(
		data,
		vector,
	)
	if err != nil {
		return nil, err
	}

	return vector, nil
}

func (d *DeviceVectorInt) Close() error {
	ret := C.DeviceVectorIntFree(
		d.pointer,
	)

	if ret != 0 {
		return ErrRawCloseDeviceVector
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
		return 0, ErrRawGetSize
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
