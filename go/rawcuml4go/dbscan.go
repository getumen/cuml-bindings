package rawcuml4go

// #cgo LDFLAGS: -lcuml4c -lcuml++ -lcuml
// #include <stdlib.h>
// #include "cuml4c/dbscan.h"
import "C"
import "errors"

var (
	ErrDBScan = errors.New("raw api: fail to dbscan")
)

// DBScan is raw api for dbscan
func DBScan(
	deviceResource *DeviceResource,
	x []float32,
	numRow int,
	numCol int,
	minPts int,
	eps float64,
	metric int,
	maxBytesPerBatch int,
	verbosity int,
	labels []int32,
) ([]int32, error) {

	if labels == nil {
		labels = make([]int32, numRow)
	}

	ret := C.DbscanFit(
		deviceResource.pointer,
		(*C.float)(&x[0]),
		(C.size_t)(numRow),
		(C.size_t)(numCol),
		(C.int)(minPts),
		(C.double)(eps),
		(C.int)(metric),
		(C.size_t)(maxBytesPerBatch),
		(C.int)(verbosity),
		(*C.int)(&labels[0]),
	)

	if ret != 0 {
		return nil, ErrDBScan
	}

	return labels, nil
}
