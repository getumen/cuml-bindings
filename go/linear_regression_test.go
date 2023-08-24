package cuml4go_test

import (
	"testing"

	cuml4go "github.com/getumen/cuml/go"
	"github.com/stretchr/testify/require"
)

func TestLinearRegression(t *testing.T) {

	target := cuml4go.NewLinearRegression(
		true,
		false,
		cuml4go.Svd,
	)

	features := csvToFloat32Array(t, "testdata/feature.csv")
	featureCol := 30
	featureRow := 114

	labels := csvToFloat32Array(t, "testdata/label.csv")

	err := target.Fit(features, featureRow, featureCol, labels)
	require.NoError(t, err)

	preds, err := target.Predict(features, featureRow, featureCol)
	require.NoError(t, err)

	require.Equal(t, len(labels), len(preds))
}

func TestRidgeRegression(t *testing.T) {
	target := cuml4go.NewRidgeRegression(
		0.5,
		true,
		false,
		cuml4go.Svd,
	)

	features := csvToFloat32Array(t, "testdata/feature.csv")
	featureCol := 30
	featureRow := 114

	labels := csvToFloat32Array(t, "testdata/label.csv")

	err := target.Fit(features, featureRow, featureCol, labels)
	require.NoError(t, err)

	preds, err := target.Predict(features, featureRow, featureCol)
	require.NoError(t, err)

	require.Equal(t, len(labels), len(preds))
}

func TestQtGLMRegression(t *testing.T) {
	target := cuml4go.NewQnGlmRegression(
		cuml4go.Normal,
		true,
		0.5,
		0.5,
		1,
	)

	features := csvToFloat32Array(t, "testdata/feature.csv")
	featureCol := 30
	featureRow := 114

	require.Equal(t, len(features), featureRow*featureCol)

	labels := csvToFloat32Array(t, "testdata/label.csv")

	loss, err := target.Fit(
		features,
		featureRow,
		featureCol,
		true,
		labels,
		1,
		nil,
		10,
		0.01,
		0.01,
		10,
		16)
	require.NoError(t, err)
	require.Greater(t, loss, float32(0.0))

	preds, err := target.Predict(features, featureRow, featureCol, true)
	require.NoError(t, err)

	require.Equal(t, len(labels), len(preds))
}
