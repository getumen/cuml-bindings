package rawcuml4go

// #cgo LDFLAGS: -lcuml4c -lcuml++ -lcuml -lcumlprims
// #include <stdlib.h>
// #include "cuml4c/kmeans.h"
import "C"
import "errors"

var (
	ErrKmeans = errors.New("raw api: fail to kmeans")
)

func Kmeans(
	x *DeviceVectorFloat,
	numRow int,
	numCol int,
	sampleWeight *DeviceVectorFloat,
	k int,
	maxIter int,
	tol float64,
	init int,
	metric int,
	seed int,
	verbosity int,
) (
	labels *DeviceVectorInt,
	centroids *DeviceVectorFloat,
	inertia float32,
	nIter int32,
	err error,
) {
	labels = NewDeviceVectorIntEmpty()
	centroids = NewDeviceVectorFloatEmpty()

	var ret C.int
	if sampleWeight == nil {
		ret = C.KmeansFit(
			x.pointer,
			(C.int)(numRow),
			(C.int)(numCol),
			nil,
			(C.int)(k),
			(C.int)(maxIter),
			(C.double)(tol),
			C.int(init),
			C.int(metric),
			(C.int)(seed),
			(C.int)(verbosity),
			&labels.pointer,
			&centroids.pointer,
			(*C.float)(&inertia),
			(*C.int)(&nIter),
		)
	} else {
		ret = C.KmeansFit(
			x.pointer,
			(C.int)(numRow),
			(C.int)(numCol),
			sampleWeight.pointer,
			(C.int)(k),
			(C.int)(maxIter),
			(C.double)(tol),
			C.int(init),
			C.int(metric),
			(C.int)(seed),
			(C.int)(verbosity),
			&labels.pointer,
			&centroids.pointer,
			(*C.float)(&inertia),
			(*C.int)(&nIter),
		)
	}

	if ret != 0 {
		err = ErrKmeans
	}

	return
}
