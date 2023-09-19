package rawcuml4go

// #cgo LDFLAGS: -lcuml4c -lcuml++ -lcuml
// #include <stdlib.h>
// #include "cuml4c/kmeans.h"
import "C"
import "errors"

var (
	ErrKmeans = errors.New("raw api: fail to kmeans")
)

func Kmeans(
	deviceResource *DeviceResource,
	x []float32,
	numRow int,
	numCol int,
	k int,
	maxIter int,
	tol float64,
	init int,
	metric int,
	seed int,
	verbosity int,
	labels []int32,
	centroids []float32,
) (
	[]int32,
	[]float32,
	float32,
	int32,
	error,
) {
	if labels == nil {
		labels = make([]int32, numRow)
	}

	if centroids == nil {
		centroids = make([]float32, k*numCol)
	}

	var inertia float32
	var nIter int32

	var ret C.int
	ret = C.KmeansFit(
		deviceResource.pointer,
		(*C.float)(&x[0]),
		(C.int)(numRow),
		(C.int)(numCol),
		(C.int)(k),
		(C.int)(maxIter),
		(C.double)(tol),
		C.int(init),
		C.int(metric),
		(C.int)(seed),
		(C.int)(verbosity),
		(*C.int)(&labels[0]),
		(*C.float)(&centroids[0]),
		(*C.float)(&inertia),
		(*C.int)(&nIter),
	)

	if ret != 0 {
		return nil, nil, 0, 0, ErrKmeans
	}

	return labels, centroids, inertia, nIter, nil
}
