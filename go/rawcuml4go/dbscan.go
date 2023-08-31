package rawcuml4go

// #cgo LDFLAGS: -lcuml4c -lcuml++ -lcuml -lcumlprims
// #include <stdlib.h>
// #include "cuml4c/dbscan.h"
import "C"
import "errors"

var (
	ErrDBScan = errors.New("raw api: fail to dbscan")
)

func DBScan(
	x *DeviceVectorFloat,
	numRow int,
	numCol int,
	minPts int,
	eps float64,
	metric int,
	maxBytesPerBatch int,
	verbosity int,
) (*DeviceVectorInt, error) {

	result := NewDeviceVectorIntEmpty()

	ret := C.DbscanFit(
		x.pointer,
		(C.size_t)(numRow),
		(C.size_t)(numCol),
		(C.int)(minPts),
		(C.double)(eps),
		(C.int)(metric),
		(C.size_t)(maxBytesPerBatch),
		(C.int)(verbosity),
		&result.pointer,
	)

	if ret != 0 {
		return nil, ErrDBScan
	}

	return result, nil
}
