package rawcuml4go

// #cgo LDFLAGS: -lcuml4c -lcuml++ -lcuml -lcumlprims
// #include <stdlib.h>
// #include "cuml4c/agglomerative_clustering.h"
import "C"
import "errors"

var (
	ErrAgglomerativeClustering = errors.New("raw api: fail to agglomerative clustering")
)

func AgglomerativeClustering(
	x *DeviceVectorFloat,
	numRow int,
	numCol int,
	pairwiseConn bool,
	metric int,
	initNumCluster int,
	numNeighbor int,
) (
	numCluster int32,
	labels *DeviceVectorInt,
	children *DeviceVectorInt,
	err error,
) {

	labels = NewDeviceVectorIntEmpty()
	children = NewDeviceVectorIntEmpty()

	ret := C.AgglomerativeClusteringFit(
		x.pointer,
		(C.ulong)(numRow),
		(C.ulong)(numCol),
		(C.bool)(pairwiseConn),
		(C.int)(metric),
		(C.int)(numNeighbor),
		(C.int)(initNumCluster),
		(*C.int)(&numCluster),
		&labels.pointer,
		&children.pointer,
	)

	if ret != 0 {
		err = ErrAgglomerativeClustering
	}

	return
}
