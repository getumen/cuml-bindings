package cuml4go

import (
	"errors"

	"github.com/getumen/cuml/go/rawcuml4go"
)

var (
	ErrAgglomerativeClustering = errors.New("fail to agglomerative clustering")
)

type AgglomerativeClustering struct {
	pairwiseConn   bool
	metric         Metric
	initNumCluster int
	numNeighbor    int
}

func NewAgglomerativeClustering(
	pairwiseConn bool,
	metric Metric,
	initNumCluster int,
	numNeighbor int,
) *AgglomerativeClustering {
	return &AgglomerativeClustering{
		pairwiseConn:   pairwiseConn,
		metric:         metric,
		initNumCluster: initNumCluster,
		numNeighbor:    numNeighbor,
	}
}

func (c *AgglomerativeClustering) Fit(
	x []float32,
	numRow int,
	numCol int,
) (numCluster int32, labels []int32, children []int32, err error) {

	dX, err := rawcuml4go.NewDeviceVectorFloat(x)

	if err != nil {
		return
	}

	numCluster, dLabels, dChildren, err := rawcuml4go.AgglomerativeClustering(
		dX,
		numRow,
		numCol,
		c.pairwiseConn,
		int(c.metric),
		c.initNumCluster,
		c.numNeighbor,
	)

	if err != nil {
		err = ErrAgglomerativeClustering
	}

	labels, err = dLabels.ToHost()
	if err != nil {
		return
	}

	children, err = dChildren.ToHost()
	if err != nil {
		return
	}

	return
}
