package rawcuml4go

// #cgo LDFLAGS: -lcuml4c -lcuml++ -lcuml -lcumlprims_mg
// #include <stdlib.h>
// #include "cuml4c/agglomerative_clustering.h"
import "C"
import "errors"

var (
	ErrAgglomerativeClustering = errors.New("raw api: fail to agglomerative clustering")
)

// AgglomerativeClustering is raw api for agglomerative clustering
func AgglomerativeClustering(
	x []float32,
	numRow int,
	numCol int,
	pairwiseConn bool,
	metric int,
	initNumCluster int,
	numNeighbor int,
	labels []int32,
	children []int32,
) (
	[]int32,
	[]int32,
	int32,
	error,
) {

	if labels == nil {
		labels = make([]int32, numRow)
	}

	if children == nil {
		children = make([]int32, (numRow-1)*2)
	}

	var numCluster int32

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
		return nil, nil, 0, ErrAgglomerativeClustering
	}

	return labels, children, numCluster, nil
}
