package rawcuml4go

// #cgo LDFLAGS: -lcuml4c -lcuml++ -lcuml -lcumlprims
// #include <stdlib.h>
// #include "cuml4c/fil.h"
import "C"
import "errors"

var (
	// ErrFILModelLoad is returned when fail to load model.
	ErrFILModelLoad = errors.New("raw api: fail to load model")
	// ErrFILModelFree is returned when fail to free model.
	ErrFILModelFree = errors.New("raw api: fail to free model")
	// ErrFILModelPredict is returned when fail to predict.
	ErrFILModelPredict = errors.New("raw api: fail to predict")
)

// FILModel is a Forest Inference Library model.
type FILModel struct {
	pointer  C.FILModelHandle
	numClass int
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
	modelType int,
	filePath string,
	algo int,
	classification bool,
	threshold float32,
	storageType int,
	blocksPerSm int,
	threadsPerTree int,
	nItems int,
) (*FILModel, error) {
	var handle C.FILModelHandle
	ret := C.FILLoadModel(
		C.int(modelType),
		C.CString(filePath),
		C.int(algo),
		C.bool(classification),
		C.float(threshold),
		C.int(storageType),
		C.int(blocksPerSm),
		C.int(threadsPerTree),
		C.int(nItems),
		&handle,
	)
	if ret != 0 {
		return nil, ErrFILModelLoad
	}

	var numClass uint64
	ret = C.FILGetNumClasses(handle, (*C.ulong)(&numClass))
	if ret != 0 {
		return nil, ErrFILModelLoad
	}

	return &FILModel{
		pointer:  handle,
		numClass: int(numClass),
	}, nil

}

// Predict returns the prediction result in device.
func (m *FILModel) Predict(
	x *DeviceVectorFloat,
	numRow int,
	outputClassProbability bool,
	preds *DeviceVectorFloat,
) (*DeviceVectorFloat, error) {

	var err error

	if preds == nil {
		var predSize int
		if outputClassProbability {
			predSize = numRow * m.numClass
		} else {
			predSize = numRow
		}

		preds, err = NewDeviceVectorFloat(predSize)
		if err != nil {
			return nil, err
		}
	}

	ret := C.FILPredict(
		m.pointer,
		x.pointer,
		(C.size_t)(numRow),
		(C.bool)(outputClassProbability),
		preds.pointer,
	)

	if ret != 0 {
		return nil, ErrFILModelPredict
	}

	return preds, nil
}

// Close frees the model.
func (m *FILModel) Close() error {
	ret := C.FILFreeModel(m.pointer)
	if ret != 0 {
		return ErrFILModelFree
	}
	return nil
}

func (m *FILModel) NumClass() int {
	return m.numClass
}
