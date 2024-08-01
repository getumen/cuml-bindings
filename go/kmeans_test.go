package cuml4go_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	cuml4go "github.com/getumen/cuml-bindings/go"
)

func TestKmeans(t *testing.T) {

	features := csvToFloat32Array(t, "../testdata/feature.csv")
	featureCol := 30
	featureRow := 114
	k := 3

	target, err := cuml4go.NewKmeans(
		k,
		10,
		0.0,
		cuml4go.KMeansPlusPlus,
		cuml4go.L2Expanded,
		42,
		cuml4go.Info,
	)
	require.NoError(t, err)

	labels, centroids, inertia, nIter, err := target.Fit(
		features,
		featureRow,
		featureCol,
		nil,
	)

	require.NoError(t, err)

	require.Equal(t, len(labels), featureRow)
	require.Equal(t, len(centroids), k*featureCol)
	require.GreaterOrEqual(t, inertia, float32(0.0))
	require.GreaterOrEqual(t, nIter, int32(0))
	require.LessOrEqual(t, nIter, int32(10))

}
