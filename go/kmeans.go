package cuml4go

import (
	"errors"

	"github.com/getumen/cuml/go/rawcuml4go"
)

var (
	ErrKmeans = errors.New("fail to kmeans")
)

type KmeansInit int

const (
	KMeansPlusPlus = iota
	Random
	Array
)

type Kmeans struct {
	k         int
	maxIter   int
	tol       float64
	init      KmeansInit
	metric    Metric
	seed      int
	verbosity LogLevel
}

func NewKmeans(
	k int,
	maxIter int,
	tol float64,
	init KmeansInit,
	metric Metric,
	seed int,
	verbosity LogLevel,
) *Kmeans {
	return &Kmeans{
		k:         k,
		maxIter:   maxIter,
		tol:       tol,
		init:      init,
		metric:    metric,
		seed:      seed,
		verbosity: verbosity,
	}
}

func (k *Kmeans) Fit(
	x []float32,
	numRow int,
	numCol int,
	sampleWeight []float32,

) (
	labels []int32,
	centroids []float32,
	inertia float32,
	nIter int32,
	err error,
) {
	dX, err := rawcuml4go.NewDeviceVectorFloatFromData(x)

	if err != nil {
		return
	}

	var dSampleWeight *rawcuml4go.DeviceVectorFloat
	if sampleWeight != nil {
		dSampleWeight, err = rawcuml4go.NewDeviceVectorFloatFromData(sampleWeight)
		if err != nil {
			return
		}
	}

	dLabels, dCentroids, inertia, nIter, err := rawcuml4go.Kmeans(
		dX,
		numRow,
		numCol,
		dSampleWeight,
		k.k,
		k.maxIter,
		k.tol,
		int(k.init),
		int(k.metric),
		k.seed,
		int(k.verbosity),
		nil,
		nil,
	)
	if err != nil {
		err = ErrKmeans
	}
	defer dLabels.Close()
	defer dCentroids.Close()

	labels, err = dLabels.ToHost()
	if err != nil {
		return
	}

	centroids, err = dCentroids.ToHost()
	if err != nil {
		return
	}

	return
}
