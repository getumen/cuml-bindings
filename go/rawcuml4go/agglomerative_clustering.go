package rawcuml4go

// #cgo LDFLAGS: -lcuml4c -lcuml++ -lcuml -lcumlprims
// #include <stdlib.h>
// #include "cuml4c/agglomerative_clustering.h"
import "C"
import "errors"

var (
	ErrAgglomerativeClustering = errors.New("raw api: fail to agglomerative clustering")
)

// AgglomerativeClustering is raw api for agglomerative clustering
func AgglomerativeClustering(
	x *DeviceVectorFloat,
	numRow int,
	numCol int,
	pairwiseConn bool,
	metric int,
	initNumCluster int,
	numNeighbor int,
	labels *DeviceVectorInt,
	children *DeviceVectorInt,
) (
	*DeviceVectorInt,
	*DeviceVectorInt,
	int32,
	error,
) {
	var err error

	if labels == nil {
		labels, err = NewDeviceVectorInt(numRow)
		if err != nil {
			return nil, nil, 0, err
		}
	}

	if children == nil {
		children, err = NewDeviceVectorInt(2 * (numRow - 1))
		if err != nil {
			return nil, nil, 0, err
		}
	}

	var numCluster int32

	ret := C.AgglomerativeClusteringFit(
		x.pointer,
		(C.ulong)(numRow),
		(C.ulong)(numCol),
		(C.bool)(pairwiseConn),
		(C.int)(metric),
		(C.int)(numNeighbor),
		(C.int)(initNumCluster),
		(*C.int)(&numCluster),
		labels.pointer,
		children.pointer,
	)

	if ret != 0 {
		return nil, nil, 0, ErrAgglomerativeClustering
	}

	return labels, children, numCluster, nil
}
