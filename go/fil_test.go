package cuml4go_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	cuml4go "github.com/getumen/cuml/go"
)

func TestFIL(t *testing.T) {

	target, err := cuml4go.NewFILModel(
		cuml4go.XGBoost,
		"testdata/xgboost.model",
		cuml4go.AlgoAuto,
		true,
		0.0,
		cuml4go.Auto,
		0,
		1,
		0)
	if err != nil {
		t.Fatal(err)
	}

	nRow := 114

	features := csvToFloat32Array(t, "testdata/feature.csv")
	expectedScores := csvToFloat32Array(t, "testdata/score-xgboost.csv")

	actual, err := target.PredictSingleClassScore(features, nRow)
	if err != nil {
		t.Fatal(err)
	}

	require.Equal(t, len(expectedScores), len(actual))
	require.InDeltaSlice(t, expectedScores, actual, 1e-4)

	defer target.Close()
}
