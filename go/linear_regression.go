package cuml4go

// #cgo LDFLAGS: -lcuml4c -lcuml++ -lcuml -lcumlprims
// #include <stdlib.h>
// #include "cuml4c/linear_regression.h"
import "C"
import "errors"

var (
	ErrLinearRegressionFit     = errors.New("fail to linear regression fit")
	ErrLinearRegressionPredict = errors.New("fail to linear regression predict")
	ErrRidgeRegressionFit      = errors.New("fail to ridge regression fit")
	ErrRidgeRegressionPredict  = errors.New("fail to ridge regression predict")
	ErrQnGlmRegressionFit      = errors.New("fail to qn glm regression fit")
	ErrQnGlmRegressionPredict  = errors.New("fail to qn glm regression predict")
)

type GlmSolverAlgo int

const (
	Svd = 0
	Eig = 1
	Qr  = 2
)

type LinearRegression struct {
	coef         []float32
	intercept    float32
	fitIntercept bool
	normalize    bool
	algo         GlmSolverAlgo
}

func NewLinearRegression(
	fitIntercept bool,
	normalize bool,
	algo GlmSolverAlgo,
) *LinearRegression {
	return &LinearRegression{
		fitIntercept: fitIntercept,
		normalize:    normalize,
		algo:         algo,
	}
}

func (m *LinearRegression) Fit(
	x []float32,
	numRow int,
	numCol int,
	labels []float32,
) error {
	m.coef = make([]float32, numCol)

	ret := C.OlsFit(
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
	x []float32,
	numRow int,
	numCol int,
) ([]float32, error) {
	result := make([]float32, numRow)

	ret := C.GemmPredict(
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
	algo         GlmSolverAlgo
}

func NewRidgeRegression(
	alpha float32,
	fitIntercept bool,
	normalize bool,
	algo GlmSolverAlgo,
) *RidgeRegression {
	return &RidgeRegression{
		alpha:        alpha,
		fitIntercept: fitIntercept,
		normalize:    normalize,
		algo:         algo,
	}
}

func (m *RidgeRegression) Fit(
	x []float32,
	numRow int,
	numCol int,
	labels []float32,
) error {
	m.coef = make([]float32, numCol)

	alpha := []float32{m.alpha}

	ret := C.RidgeFit(
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
	x []float32,
	numRow int,
	numCol int,
) ([]float32, error) {
	result := make([]float32, numRow)

	ret := C.GemmPredict(
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

type LossType int

const (
	Logistic     = 0
	Normal       = 1
	Multinomiral = 2
)

type QnGlmRegression struct {
	coef         []float32
	numClass     int
	lossType     LossType
	fitIntercept bool
	l1           float32
	l2           float32
	verbosity    int
}

func NewQnGlmRegression(
	lossType LossType,
	fitIntercept bool,
	l1 float32,
	l2 float32,
	verbosity int,
) *QnGlmRegression {
	return &QnGlmRegression{
		lossType:     lossType,
		fitIntercept: fitIntercept,
		l1:           l1,
		l2:           l2,
		verbosity:    verbosity,
	}
}

func (m *QnGlmRegression) Fit(
	x []float32,
	numRow int,
	numCol int,
	xColMajor bool,
	labels []float32,
	numClass int,
	sampleWeight []float32,
	maxIter int,
	gradTol float32,
	changeTol float32,
	linesearchMaxIter int,
	lgbfMemory int,
) (float32, error) {
	var intercept int
	if m.fitIntercept {
		intercept = 1
	}

	m.coef = make([]float32, (numCol+intercept)*numClass)
	var loss float32
	var nIter int32

	var sampleWeightPtr *C.float
	if sampleWeight != nil {
		sampleWeightPtr = (*C.float)(&sampleWeight[0])
	}

	ret := C.QnFit(
		(*C.float)(&x[0]),
		(C.ulong)(numRow),
		(C.ulong)(numCol),
		(C.bool)(xColMajor),
		(*C.float)(&labels[0]),
		(C.ulong)(numClass),
		(C.int)(m.lossType),
		sampleWeightPtr,
		(C.bool)(m.fitIntercept),
		(C.float)(m.l1),
		(C.float)(m.l2),
		(C.int)(maxIter),
		(C.float)(gradTol),
		(C.float)(changeTol),
		(C.int)(linesearchMaxIter),
		(C.int)(lgbfMemory),
		(C.int)(m.verbosity),
		(*C.float)(&m.coef[0]),
		(*C.float)(&loss),
		(*C.int)(&nIter),
	)

	if ret != 0 {
		return 0, ErrLinearRegressionFit
	}

	return loss, nil
}

func (m *QnGlmRegression) FitSparse(
	values []float32,
	indices []int32,
	header []int32,
	numRow int,
	numCol int,
	numNonZero int,
	labels []float32,
	numClass int,
	sampleWeight []float32,
	maxIter int,
	gradTol float32,
	changeTol float32,
	linesearchMaxIter int,
	lgbfMemory int,
) (float32, error) {
	var intercept int
	if m.fitIntercept {
		intercept = 1
	}

	m.coef = make([]float32, (numCol+intercept)*numClass)
	var loss float32
	var nIter int32

	var sampleWeightPtr *C.float
	if sampleWeight != nil {
		sampleWeightPtr = (*C.float)(&sampleWeight[0])
	}

	ret := C.QnFitSparse(
		(*C.float)(&values[0]),
		(*C.int)(&indices[0]),
		(*C.int)(&header[0]),
		(C.ulong)(numRow),
		(C.ulong)(numCol),
		(C.ulong)(numNonZero),
		(*C.float)(&labels[0]),
		(C.ulong)(numClass),
		(C.int)(m.lossType),
		sampleWeightPtr,
		(C.bool)(m.fitIntercept),
		(C.float)(m.l1),
		(C.float)(m.l2),
		(C.int)(maxIter),
		(C.float)(gradTol),
		(C.float)(changeTol),
		(C.int)(linesearchMaxIter),
		(C.int)(lgbfMemory),
		(C.int)(m.verbosity),
		(*C.float)(&m.coef[0]),
		(*C.float)(&loss),
		(*C.int)(&nIter),
	)

	if ret != 0 {
		return 0, ErrQnGlmRegressionFit
	}

	return loss, nil
}

func (m *QnGlmRegression) DecisionFunction(
	x []float32,
	numRow int,
	numCol int,
	xColMajor bool,
) ([]float32, error) {
	var intercept int
	if m.fitIntercept {
		intercept = 1
	}

	result := make([]float32, (numCol+intercept)*m.numClass)

	ret := C.QnDecisionFunction(
		(*C.float)(&x[0]),
		(C.bool)(xColMajor),
		(C.ulong)(numRow),
		(C.ulong)(numCol),
		(C.ulong)(m.numClass),
		(C.bool)(m.fitIntercept),
		(*C.float)(&m.coef[0]),
		(C.int)(m.lossType),
		(*C.float)(&result[0]),
	)

	if ret != 0 {
		return nil, ErrQnGlmRegressionPredict
	}

	return result, nil
}

func (m *QnGlmRegression) DecisionFunctionSparse(
	values []float32,
	indices []int32,
	header []int32,
	numRow int,
	numCol int,
	numNonZero int,
) ([]float32, error) {
	var intercept int
	if m.fitIntercept {
		intercept = 1
	}
	result := make([]float32, (numCol+intercept)*m.numClass)

	ret := C.QnDecisionFunctionSparse(
		(*C.float)(&values[0]),
		(*C.int)(&indices[0]),
		(*C.int)(&header[0]),
		(C.ulong)(numRow),
		(C.ulong)(numCol),
		(C.ulong)(numNonZero),
		(C.ulong)(m.numClass),
		(C.bool)(m.fitIntercept),
		(*C.float)(&m.coef[0]),
		(C.int)(m.lossType),
		(*C.float)(&result[0]),
	)

	if ret != 0 {
		return nil, ErrQnGlmRegressionPredict
	}

	return result, nil
}

func (m *QnGlmRegression) Predict(
	x []float32,
	numRow int,
	numCol int,
	xColMajor bool,
) ([]float32, error) {
	result := make([]float32, numRow)

	ret := C.QnPredict(
		(*C.float)(&x[0]),
		(C.bool)(xColMajor),
		(C.ulong)(numRow),
		(C.ulong)(numCol),
		(C.ulong)(m.numClass),
		(C.bool)(m.fitIntercept),
		(*C.float)(&m.coef[0]),
		(C.int)(m.lossType),
		(*C.float)(&result[0]),
	)

	if ret != 0 {
		return nil, ErrQnGlmRegressionPredict
	}

	return result, nil
}

func (m *QnGlmRegression) PredictSparse(
	values []float32,
	indices []int32,
	header []int32,
	numRow int,
	numCol int,
	numNonZero int,
) ([]float32, error) {
	result := make([]float32, numRow)

	ret := C.QnPredictSparse(
		(*C.float)(&values[0]),
		(*C.int)(&indices[0]),
		(*C.int)(&header[0]),
		(C.ulong)(numRow),
		(C.ulong)(numCol),
		(C.ulong)(numNonZero),
		(C.ulong)(m.numClass),
		(C.bool)(m.fitIntercept),
		(*C.float)(&m.coef[0]),
		(C.int)(m.lossType),
		(*C.float)(&result[0]),
	)

	if ret != 0 {
		return nil, ErrQnGlmRegressionPredict
	}

	return result, nil
}

func (m *QnGlmRegression) GetParams() []float32 {
	return m.coef
}

func (m *QnGlmRegression) SetParams(coef []float32) {
	m.coef = coef
}
