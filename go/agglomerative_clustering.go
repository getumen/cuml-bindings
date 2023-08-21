package cuml4go

// #cgo LDFLAGS: -lcuml4c -lcuml++ -lcuml -lcumlprims
// #include <stdlib.h>
// #include "cuml4c/agglomerative_clustering.h"
import "C"

import "errors"

var (
	ErrAgglomerativeClustering = errors.New("fail to agglomerative clustering")
)

func AgglomerativeClustering(
	x []float32,
	numRow int,
	numCol int,
	pairwiseConn bool,
	metric Metric,
	initNumCluster int,
	numNeighbor int,
) (numCluster int32, labels []int32, children []int32, err error) {
	labels = make([]int32, numRow)
	children = make([]int32, (numRow-1)*2)

	ret := C.AgglomerativeClusteringFit(
		(*C.float)(&x[0]),
		(C.ulong)(numRow),
		(C.ulong)(numCol),
		(C.bool)(pairwiseConn),
		(C.int)(metric),
		(C.int)(numNeighbor),
		(C.int)(initNumCluster),
		(*C.int)(&numCluster),
		(*C.int)(&labels[0]),
		(*C.int)(&children[0]),
	)

	if ret != 0 {
		err = ErrAgglomerativeClustering
	}

	return
}
