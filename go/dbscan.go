package cuml4go

// #cgo LDFLAGS: -lcuml4c -lcuml++ -lcuml -lcumlprims
// #include <stdlib.h>
// #include "cuml4c/dbscan.h"
import "C"
import "errors"

var (
	ErrDBScan = errors.New("fail to dbscan")
)

func DBScan(
	x []float32,
	numRow int,
	numCol int,
	minPts int,
	eps float64,
	maxBytesPerBatch int,
	verbosity int,
) ([]int32, error) {

	result := make([]int32, numRow)

	ret := C.DbscanFit(
		(*C.float)(&x[0]),
		(C.size_t)(numRow),
		(C.size_t)(numCol),
		(C.int)(minPts),
		(C.double)(eps),
		(C.size_t)(maxBytesPerBatch),
		(C.int)(verbosity),
		(*C.int)(&result[0]),
	)

	if ret != 0 {
		return nil, ErrDBScan
	}

	return result, nil
}
