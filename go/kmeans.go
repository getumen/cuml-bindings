package cuml4go

// #cgo LDFLAGS: -lcuml4c -lcuml++ -lcuml -lcumlprims
// #include <stdlib.h>
// #include "cuml4c/kmeans.h"
import "C"
import "errors"

var (
	ErrKmeans = errors.New("fail to kmeans")
)

type KmeansInit int

const (
	KMeansPlusPlus = iota
	Random
	Array
)

func Kmeans(
	x []float32,
	numRow int,
	numCol int,
	sampleWeight []float32,
	k int,
	maxIter int,
	tol float64,
	init KmeansInit,
	seed int,
	verbosity int,
) (
	labels []int32,
	centroids []float32,
	inertia float32,
	nIter int32,
	err error,
) {
	labels = make([]int32, numRow)
	centroids = make([]float32, k*numCol)

	var ret C.int
	if sampleWeight == nil {
		ret = C.KmeansFit(
			(*C.float)(&x[0]),
			(C.int)(numRow),
			(C.int)(numCol),
			(*C.float)(nil),
			(C.int)(k),
			(C.int)(maxIter),
			(C.double)(tol),
			C.int(init),
			(C.int)(seed),
			(C.int)(verbosity),
			(*C.int)(&labels[0]),
			(*C.float)(&centroids[0]),
			(*C.float)(&inertia),
			(*C.int)(&nIter),
		)
	} else {
		ret = C.KmeansFit(
			(*C.float)(&x[0]),
			(C.int)(numRow),
			(C.int)(numCol),
			(*C.float)(&sampleWeight[0]),
			(C.int)(k),
			(C.int)(maxIter),
			(C.double)(tol),
			C.int(init),
			(C.int)(seed),
			(C.int)(verbosity),
			(*C.int)(&labels[0]),
			(*C.float)(&centroids[0]),
			(*C.float)(&inertia),
			(*C.int)(&nIter),
		)
	}

	if ret != 0 {
		err = ErrKmeans
	}

	return
}
