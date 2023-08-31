package rawcuml4go

// #cgo LDFLAGS: -lcuml4c -lcuml++ -lcuml -lcumlprims
// #include <stdlib.h>
// #include "cuml4c/dbscan.h"
import "C"
import "errors"

var (
	ErrDBScan = errors.New("raw api: fail to dbscan")
)

// DBScan is raw api for dbscan
func DBScan(
	x *DeviceVectorFloat,
	numRow int,
	numCol int,
	minPts int,
	eps float64,
	metric int,
	maxBytesPerBatch int,
	verbosity int,
	labels *DeviceVectorInt,
) (*DeviceVectorInt, error) {
	var err error

	if labels == nil {
		labels, err = NewDeviceVectorInt(numRow)
		if err != nil {
			return nil, err
		}
	}

	ret := C.DbscanFit(
		x.pointer,
		(C.size_t)(numRow),
		(C.size_t)(numCol),
		(C.int)(minPts),
		(C.double)(eps),
		(C.int)(metric),
		(C.size_t)(maxBytesPerBatch),
		(C.int)(verbosity),
		labels.pointer,
	)

	if ret != 0 {
		return nil, ErrDBScan
	}

	return labels, nil
}
