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
	labels *DeviceVectorInt,
	centroids *DeviceVectorFloat,
) (
	*DeviceVectorInt,
	*DeviceVectorFloat,
	float32,
	int32,
	error,
) {
	var err error
	if labels == nil {
		labels, err = NewDeviceVectorInt(numRow)
		if err != nil {
			return nil, nil, 0, 0, err
		}
	}

	if centroids == nil {
		centroids, err = NewDeviceVectorFloat(k * numCol)
		if err != nil {
			return nil, nil, 0, 0, err
		}
	}

	var inertia float32
	var nIter int32

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
			labels.pointer,
			centroids.pointer,
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
			labels.pointer,
			centroids.pointer,
			(*C.float)(&inertia),
			(*C.int)(&nIter),
		)
	}

	if ret != 0 {
		return nil, nil, 0, 0, ErrKmeans
	}

	return labels, centroids, inertia, nIter, nil
}
