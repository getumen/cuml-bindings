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

	labels, centroids, inertia, nIter, err = rawcuml4go.Kmeans(
		x,
		numRow,
		numCol,
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

	return
}
