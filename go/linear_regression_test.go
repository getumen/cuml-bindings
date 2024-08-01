package cuml4go_test

import (
	"testing"

	cuml4go "github.com/getumen/cuml-bindings/go"
	"github.com/stretchr/testify/require"
)

func TestLinearRegression(t *testing.T) {

	target, err := cuml4go.NewLinearRegression(
		true,
		false,
		cuml4go.Svd,
	)
	require.NoError(t, err)
	defer target.Close()

	features := csvToFloat32Array(t, "../testdata/feature.csv")
	featureCol := 30
	featureRow := 114

	labels := csvToFloat32Array(t, "../testdata/label.csv")

	err = target.Fit(features, featureRow, featureCol, labels)
	require.NoError(t, err)

	preds, err := target.Predict(features, featureRow, featureCol, nil)
	require.NoError(t, err)

	require.Equal(t, len(labels), len(preds))
}

func TestRidgeRegression(t *testing.T) {
	target, err := cuml4go.NewRidgeRegression(
		0.5,
		true,
		false,
		cuml4go.Svd,
	)
	require.NoError(t, err)
	defer target.Close()

	features := csvToFloat32Array(t, "../testdata/feature.csv")
	featureCol := 30
	featureRow := 114

	labels := csvToFloat32Array(t, "../testdata/label.csv")

	err = target.Fit(features, featureRow, featureCol, labels)
	require.NoError(t, err)

	preds, err := target.Predict(features, featureRow, featureCol, nil)
	require.NoError(t, err)

	require.Equal(t, len(labels), len(preds))
}
