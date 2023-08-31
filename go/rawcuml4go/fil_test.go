package rawcuml4go_test

import (
	"testing"

	cuml4go "github.com/getumen/cuml/go"
	"github.com/getumen/cuml/go/rawcuml4go"
	"github.com/stretchr/testify/require"
)

func TestFIL(t *testing.T) {

	target, err := rawcuml4go.NewFILModel(
		int(cuml4go.XGBoost),
		"../testdata/xgboost.model",
		int(cuml4go.AlgoAuto),
		true,
		0.0,
		int(cuml4go.Dense),
		0,
		1,
		0)
	require.NoError(t, err)

	nRow := 114
	numClass := 2

	features := csvToFloat32Array(t, "../testdata/feature.csv")
	expectedScores := csvToFloat32Array(t, "../testdata/score-xgboost.csv")

	dFeatire, err := rawcuml4go.NewDeviceVectorFloat(features)
	require.NoError(t, err)

	size, err := dFeatire.GetSize()
	require.NoError(t, err)

	require.Equal(t, len(features), size)

	actual, err := target.Predict(dFeatire, nRow, true)
	if err != nil {
		t.Fatal(err)
	}

	actualSize, err := actual.GetSize()
	require.NoError(t, err)

	require.Equal(t, numClass*len(expectedScores), actualSize)

	defer target.Close()
}
