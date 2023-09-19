package rawcuml4go

// #cgo LDFLAGS: -lcuml4c -lcuml++ -lcuml
// #include <stdlib.h>
// #include "cuml4c/linear_regression.h"
import "C"
import (
	"errors"
)

var (
	ErrLinearRegressionFit     = errors.New("raw api: fail to linear regression fit")
	ErrLinearRegressionPredict = errors.New("raw api: fail to linear regression predict")
	ErrRidgeRegressionFit      = errors.New("raw api: fail to ridge regression fit")
	ErrRidgeRegressionPredict  = errors.New("raw api: fail to ridge regression predict")
)

type LinearRegression struct {
	coef         []float32
	intercept    float32
	fitIntercept bool
	normalize    bool
	algo         int
}

func NewLinearRegression(
	fitIntercept bool,
	normalize bool,
	algo int,
) *LinearRegression {
	return &LinearRegression{
		fitIntercept: fitIntercept,
		normalize:    normalize,
		algo:         algo,
	}
}

func (m *LinearRegression) Fit(
	deviceResource *DeviceResource,
	x []float32,
	numRow int,
	numCol int,
	labels []float32,
) error {
	m.coef = make([]float32, numCol)

	ret := C.OlsFit(
		deviceResource.pointer,
		(*C.float)(&x[0]),
		(C.ulong)(numRow),
		(C.ulong)(numCol),
		(*C.float)(&labels[0]),
		(C.bool)(m.fitIntercept),
		(C.bool)(m.normalize),
		(C.int)(m.algo),
		(*C.float)(&m.coef[0]),
		(*C.float)(&m.intercept),
	)

	if ret != 0 {
		return ErrLinearRegressionFit
	}

	return nil
}

func (m *LinearRegression) Predict(
	deviceResource *DeviceResource,
	x []float32,
	numRow int,
	numCol int,
	result []float32,
) ([]float32, error) {
	if result == nil {
		result = make([]float32, numRow)
	}

	ret := C.GemmPredict(
		deviceResource.pointer,
		(*C.float)(&x[0]),
		(C.ulong)(numRow),
		(C.ulong)(numCol),
		(*C.float)(&m.coef[0]),
		(C.float)(m.intercept),
		(*C.float)(&result[0]),
	)

	if ret != 0 {
		return nil, ErrLinearRegressionPredict
	}

	return result, nil
}

func (m *LinearRegression) GetParams() []float32 {
	return m.coef
}

func (m *LinearRegression) SetParams(coef []float32) {
	m.coef = coef
}

type RidgeRegression struct {
	coef         []float32
	intercept    float32
	alpha        float32
	fitIntercept bool
	normalize    bool
	algo         int
}

func NewRidgeRegression(
	alpha float32,
	fitIntercept bool,
	normalize bool,
	algo int,
) *RidgeRegression {
	return &RidgeRegression{
		alpha:        alpha,
		fitIntercept: fitIntercept,
		normalize:    normalize,
		algo:         algo,
	}
}

func (m *RidgeRegression) Fit(
	deviceResource *DeviceResource,
	x []float32,
	numRow int,
	numCol int,
	labels []float32,
) error {
	m.coef = make([]float32, numCol)

	alpha := []float32{m.alpha}

	ret := C.RidgeFit(
		deviceResource.pointer,
		(*C.float)(&x[0]),
		(C.ulong)(numRow),
		(C.ulong)(numCol),
		(*C.float)(&labels[0]),
		(*C.float)(&alpha[0]),
		(C.ulong)(len(alpha)),
		(C.bool)(m.fitIntercept),
		(C.bool)(m.normalize),
		(C.int)(m.algo),
		(*C.float)(&m.coef[0]),
		(*C.float)(&m.intercept),
	)

	if ret != 0 {
		return ErrLinearRegressionFit
	}

	return nil
}

func (m *RidgeRegression) Predict(
	deviceResource *DeviceResource,
	x []float32,
	numRow int,
	numCol int,
	result []float32,
) ([]float32, error) {
	if result == nil {
		result = make([]float32, numRow)
	}

	ret := C.GemmPredict(
		deviceResource.pointer,
		(*C.float)(&x[0]),
		(C.ulong)(numRow),
		(C.ulong)(numCol),
		(*C.float)(&m.coef[0]),
		(C.float)(m.intercept),
		(*C.float)(&result[0]),
	)

	if ret != 0 {
		return nil, ErrLinearRegressionPredict
	}

	return result, nil
}

func (m *RidgeRegression) GetParams() []float32 {
	return m.coef
}

func (m *RidgeRegression) SetParams(coef []float32) {
	m.coef = coef
}
