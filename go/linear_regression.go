package cuml4go

import (
	"github.com/getumen/cuml/go/rawcuml4go"
)

type GlmSolverAlgo int

const (
	Svd = 0
	Eig = 1
	Qr  = 2
)

type LinearRegression struct {
	deviceResource *rawcuml4go.DeviceResource
	raw            *rawcuml4go.LinearRegression
}

func NewLinearRegression(
	fitIntercept bool,
	normalize bool,
	algo GlmSolverAlgo,
) (*LinearRegression, error) {
	deviceResource, err := rawcuml4go.NewDeviceResource()

	if err != nil {
		return nil, err
	}

	raw := rawcuml4go.NewLinearRegression(
		fitIntercept,
		normalize,
		int(algo),
	)

	return &LinearRegression{
		deviceResource: deviceResource,
		raw:            raw,
	}, nil
}

func (m *LinearRegression) Fit(
	x []float32,
	numRow int,
	numCol int,
	labels []float32,
) error {
	return m.raw.Fit(
		m.deviceResource,
		x,
		numRow,
		numCol,
		labels,
	)
}

func (m *LinearRegression) Predict(
	x []float32,
	numRow int,
	numCol int,
	result []float32,
) ([]float32, error) {

	return m.raw.Predict(
		m.deviceResource,
		x,
		numRow,
		numCol,
		result,
	)
}

func (m *LinearRegression) GetParams() []float32 {
	return m.raw.GetParams()
}

func (m *LinearRegression) SetParams(coef []float32) {
	m.raw.SetParams(coef)
}

func (m *LinearRegression) Close() error {
	return m.deviceResource.Close()
}

type RidgeRegression struct {
	deviceResource *rawcuml4go.DeviceResource
	raw            *rawcuml4go.RidgeRegression
}

func NewRidgeRegression(
	alpha float32,
	fitIntercept bool,
	normalize bool,
	algo GlmSolverAlgo,
) (*RidgeRegression, error) {
	deviceResource, err := rawcuml4go.NewDeviceResource()

	if err != nil {
		return nil, err
	}

	raw := rawcuml4go.NewRidgeRegression(
		alpha,
		fitIntercept,
		normalize,
		int(algo),
	)

	return &RidgeRegression{
		deviceResource: deviceResource,
		raw:            raw,
	}, nil
}

func (m *RidgeRegression) Fit(
	x []float32,
	numRow int,
	numCol int,
	labels []float32,
) error {
	return m.raw.Fit(
		m.deviceResource,
		x,
		numRow,
		numCol,
		labels,
	)
}

func (m *RidgeRegression) Predict(
	x []float32,
	numRow int,
	numCol int,
	result []float32,
) ([]float32, error) {
	return m.raw.Predict(
		m.deviceResource,
		x,
		numRow,
		numCol,
		result,
	)
}

func (m *RidgeRegression) GetParams() []float32 {
	return m.raw.GetParams()
}

func (m *RidgeRegression) SetParams(coef []float32) {
	m.raw.SetParams(coef)
}

func (m *RidgeRegression) Close() error {
	return m.deviceResource.Close()
}
