package rawcuml4go

// #cgo LDFLAGS: -lcuml4c -lcuml++ -lcuml
// #include <stdlib.h>
// #include "cuml4c/memory_resource.h"
import "C"
import "errors"

var (
	ErrGetDeviceMemoryResource   = errors.New("raw api: fail to get device memory resource")
	ErrResetDeviceMemoryResource = errors.New("raw api: fail to reset device memory resource")
)

type MemoryResource struct {
	pointer      C.DeviceMemoryResource
	resourceType int
}

func (m *MemoryResource) Close() error {
	ret := C.ResetMemoryResource(m.pointer, (C.int)(m.resourceType))
	if ret != 0 {
		return ErrResetDeviceMemoryResource
	}

	return nil
}

func UsePoolMemoryResource(
	initialPoolSize uint64,
	maximumPoolSize uint64,
) (*MemoryResource, error) {
	var pointer C.DeviceMemoryResource
	ret := C.UsePoolMemoryResource(
		(C.size_t)(initialPoolSize),
		(C.size_t)(maximumPoolSize),
		&pointer,
	)
	if ret != 0 {
		return nil, ErrGetDeviceMemoryResource
	}

	return &MemoryResource{
		pointer:      pointer,
		resourceType: 0,
	}, nil
}

func UseBinningMemoryResource(
	minSizeExponent uint8,
	maxSizeExponent uint8,
) (*MemoryResource, error) {
	var pointer C.DeviceMemoryResource
	ret := C.UseBinningMemoryResource(
		(C.schar)(minSizeExponent),
		(C.schar)(minSizeExponent),
		&pointer,
	)
	if ret != 0 {
		return nil, ErrGetDeviceMemoryResource
	}

	return &MemoryResource{
		pointer:      pointer,
		resourceType: 1,
	}, nil
}

func UseArenaMemoryResource() (
	*MemoryResource,
	error,
) {
	var pointer C.DeviceMemoryResource
	ret := C.UseArenaMemoryResource(&pointer)
	if ret != 0 {
		return nil, ErrGetDeviceMemoryResource
	}

	return &MemoryResource{
		pointer:      pointer,
		resourceType: 2,
	}, nil
}
