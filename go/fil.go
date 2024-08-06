package cuml4go

import (
	"errors"

	"github.com/getumen/cuml-bindings/go/rawcuml4go"
	"go.uber.org/multierr"
)

var (
	// ErrFILModelLoad is returned when fail to load model.
	ErrFILModelLoad = errors.New("fail to load model")
	// ErrFILModelFree is returned when fail to free model.
	ErrFILModelFree = errors.New("fail to free model")
	// ErrFILModelPredict is returned when fail to predict.
	ErrFILModelPredict = errors.New("fail to predict")
)

// FILModelType is the type of the forest.
type FILModelType int

const (
	// XGBoost xgboost model (binary model file)
	XGBoost FILModelType = iota
	// XGBoostJSON xgboost model (json model file)
	XGBoostJSON
	// LightGBM lighgbm model (binary model file)
	LightGBM
)

// FILInferenceAlgorithm is the inference algorithm.
type FILInferenceAlgorithm int

const (
	// AlgoAuto choose the algorithm automatically; currently chooses NAIVE for sparse forests
	//  and BatchTreeReorg for dense ones
	AlgoAuto FILInferenceAlgorithm = iota
	// Naive naive algorithm: 1 thread block predicts 1 row; the row is cached in
	//  shared memory, and the trees are distributed cyclically between threads
	Naive
	// TreeReorg tree reorg algorithm: same as naive, but the tree nodes are rearranged
	//  into a more coalescing-friendly layout: for every node position,
	//  nodes of all trees at that position are stored next to each other
	TreeReorg
	// BatchTreeReorg batch tree reorg algorithm: same as tree reorg, but predictions multiple rows (up to 4)
	//  in a single thread block
	BatchTreeReorg
)

// FILStorageType is the storage type of the forest.
type FILStorageType int

const (
	// Auto decide automatically; currently always builds dense forests
	Auto FILStorageType = iota
	// Dense import the forest as dense
	Dense
	// Sparse import the forest as sparse (currently always with 16-byte nodes)
	Sparse
	// Sparse8 (experimental) import the forest as sparse with 8-byte nodes; can fail if
	//  8-byte nodes are not enough to store the forest, e.g. there are too many
	//  nodes in a tree or too many features; note that the number of bits used to
	//  store the child or feature index can change in the future; this can affect
	//  whether a particular forest can be imported as SPARSE8 */
	Sparse8
)

// FILModel is a Forest Inference Library model.
type FILModel struct {
	raw            *rawcuml4go.FILModel
	deviceResource *rawcuml4go.DeviceResource
}

// NewFILModel
// algo is the inference algorithm.
// threshold may be used for thresholding if classification == true,
// and is ignored otherwise. threshold is ignored if leaves store
// vectorized class labels. in that case, a class with most votes
// is returned regardless of the absolute vote count.
// blocksPerSm if nonzero, works as a limit to improve cache hit rate for larger forests
// suggested values (if nonzero) are from 2 to 7.
// if zero, launches ceildiv(num_rows, NITEMS) blocks.
// threadsPerTree determines how many threads work on a single tree at once inside a block
// can only be a power of 2
// nItems is how many input samples (items) any thread processes. If 0 is given,
// choose most (up to 4) that fit into shared memory.
func NewFILModel(
	modelType FILModelType,
	filePath string,
	algo FILInferenceAlgorithm,
	classification bool,
	threshold float32,
	storageType FILStorageType,
	blocksPerSm int,
	threadsPerTree int,
	nItems int,
) (*FILModel, error) {
	deviceResource, err := rawcuml4go.NewDeviceResource()

	if err != nil {
		return nil, err
	}

	raw, err := rawcuml4go.NewFILModel(
		deviceResource,
		int(modelType),
		filePath,
		int(algo),
		classification,
		threshold,
		int(storageType),
		blocksPerSm,
		threadsPerTree,
		nItems,
	)

	if err != nil {
		return nil, err
	}

	return &FILModel{
		raw:            raw,
		deviceResource: deviceResource,
	}, nil
}

// Predict returns the prediction result.
// result is a float array of size num_row * num_class if output_class_probability is true,
// or num_row otherwise.
// given a row r and class c, the probability of r belonging to c is stored in result[r * num_class + c].
func (m *FILModel) Predict(
	x []float32,
	numRow int,
	outputClassProbability bool) ([]float32, error) {

	preds, err := m.raw.Predict(x, numRow, outputClassProbability, nil)
	if err != nil {
		return nil, err
	}
	return preds, nil
}

// PredictSingleClassScore returns the prediction result of the 1 class of {0,1} classification.
func (m *FILModel) PredictSingleClassScore(
	x []float32,
	numRow int,
) ([]float32, error) {
	resultRaw, err := m.Predict(x, numRow, true)
	if err != nil {
		return nil, err
	}

	result := make([]float32, numRow)
	for i := 0; i < numRow; i++ {
		result[i] = resultRaw[i*2+1]
	}
	return result, nil
}

// Close frees the model.
func (m *FILModel) Close() error {
	var err error
	err = multierr.Append(err, m.raw.Close())
	err = multierr.Append(err, m.deviceResource.Close())
	return err
}

