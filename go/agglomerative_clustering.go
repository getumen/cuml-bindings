package cuml4go

import (
	"errors"

	"github.com/getumen/cuml-bindings/go/rawcuml4go"
)

var (
	ErrAgglomerativeClustering = errors.New("fail to agglomerative clustering")
)

type AgglomerativeClustering struct {
	deviceResource *rawcuml4go.DeviceResource
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
) (*AgglomerativeClustering, error) {
	deviceResource, err := rawcuml4go.NewDeviceResource()

	if err != nil {
		return nil, err
	}

	return &AgglomerativeClustering{
		deviceResource: deviceResource,
		pairwiseConn:   pairwiseConn,
		metric:         metric,
		initNumCluster: initNumCluster,
		numNeighbor:    numNeighbor,
	}, nil
}

// Fit returns agglomerative clustering result
// output:
// labels: cluster labels
// children: children of each node
// numCluster: number of cluster
// error: error
func (c *AgglomerativeClustering) Fit(
	x []float32,
	numRow int,
	numCol int,
) ([]int32, []int32, int32, error) {

	labels, children, numCluster, err := rawcuml4go.AgglomerativeClustering(
		c.deviceResource,
		x,
		numRow,
		numCol,
		c.pairwiseConn,
		int(c.metric),
		c.initNumCluster,
		c.numNeighbor,
		nil,
		nil,
	)
	if err != nil {
		return nil, nil, 0, err
	}

	return labels, children, numCluster, nil
}

func (c *AgglomerativeClustering) Close() error {
	return c.deviceResource.Close()
}
