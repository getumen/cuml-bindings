package rawcuml4go_test

import (
	"testing"

	cuml4go "github.com/getumen/cuml/go"
	"github.com/getumen/cuml/go/rawcuml4go"
	"github.com/stretchr/testify/require"
)

func TestLinearRegression(t *testing.T) {
	deviceResource, err := rawcuml4go.NewDeviceResource()
	require.NoError(t, err)
	defer deviceResource.Close()

	target := rawcuml4go.NewLinearRegression(
		true,
		false,
		cuml4go.Svd,
	)

	features := csvToFloat32Array(t, "../../testdata/feature.csv")
	featureCol := 30
	featureRow := 114

	labels := csvToFloat32Array(t, "../../testdata/label.csv")

	err = target.Fit(deviceResource, features, featureRow, featureCol, labels)
	require.NoError(t, err)

	preds, err := target.Predict(deviceResource, features, featureRow, featureCol, nil)
	require.NoError(t, err)

	require.Equal(t, len(labels), len(preds))
}

func TestRidgeRegression(t *testing.T) {
	deviceResource, err := rawcuml4go.NewDeviceResource()
	require.NoError(t, err)
	defer deviceResource.Close()

	target := rawcuml4go.NewRidgeRegression(
		0.5,
		true,
		false,
		cuml4go.Svd,
	)

	features := csvToFloat32Array(t, "../../testdata/feature.csv")
	featureCol := 30
	featureRow := 114

	labels := csvToFloat32Array(t, "../../testdata/label.csv")

	err = target.Fit(deviceResource, features, featureRow, featureCol, labels)
	require.NoError(t, err)

	preds, err := target.Predict(deviceResource, features, featureRow, featureCol, nil)
	require.NoError(t, err)

	require.Equal(t, len(labels), len(preds))
}
