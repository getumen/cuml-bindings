package cuml4go_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	cuml4go "github.com/getumen/cuml/go"
)

func TestKmeans(t *testing.T) {

	features := csvToFloat32Array(t, "testdata/feature.csv")
	featureCol := 30
	featureRow := 114
	k := 3

	labels, centroids, inertia, nIter, err := cuml4go.Kmeans(
		features,
		featureRow,
		featureCol,
		nil,
		k,
		10,
		0.0,
		cuml4go.KMeansPlusPlus,
		42,
		0,
	)

	require.NoError(t, err)

	require.Equal(t, len(labels), featureRow)
	require.Equal(t, len(centroids), k*featureCol)
	require.GreaterOrEqual(t, inertia, float32(0.0))
	require.GreaterOrEqual(t, nIter, int32(0))
	require.LessOrEqual(t, nIter, int32(10))

}
