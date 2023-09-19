package cuml4go

import (
	"errors"

	"github.com/getumen/cuml/go/rawcuml4go"
)

var (
	ErrDBScan = errors.New("fail to dbscan")
)

type DBScan struct {
	deviceResource   *rawcuml4go.DeviceResource
	minPts           int
	eps              float64
	metric           Metric
	maxBytesPerBatch int
	verbosity        LogLevel
}

func NewDBScan(
	minPts int,
	eps float64,
	metric Metric,
	maxBytesPerBatch int,
	verbosity LogLevel,
) (*DBScan, error) {
	deviceResource, err := rawcuml4go.NewDeviceResource()

	if err != nil {
		return nil, err
	}

	return &DBScan{
		deviceResource:   deviceResource,
		minPts:           minPts,
		eps:              eps,
		metric:           metric,
		maxBytesPerBatch: maxBytesPerBatch,
		verbosity:        verbosity,
	}, nil
}

func (d *DBScan) Fit(
	x []float32,
	numRow int,
	numCol int,
) ([]int32, error) {

	labels, err := rawcuml4go.DBScan(
		d.deviceResource,
		x,
		numRow,
		numCol,
		d.minPts,
		d.eps,
		int(d.metric),
		d.maxBytesPerBatch,
		int(d.verbosity),
		nil,
	)
	if err != nil {
		return nil, err
	}

	if err != nil {
		return nil, err
	}

	return labels, nil
}

func (d *DBScan) Close() error {
	return d.deviceResource.Close()
}
