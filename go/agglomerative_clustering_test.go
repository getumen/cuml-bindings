package cuml4go_test

import (
	"testing"

	cuml4go "github.com/getumen/cuml/go"
	"github.com/stretchr/testify/require"
)

func TestAgglomerativeClustering(t *testing.T) {
	features := csvToFloat32Array(t, "../testdata/feature.csv")
	featureCol := 30
	featureRow := 114

	target, err := cuml4go.NewAgglomerativeClustering(
		false,
		cuml4go.L2SqrtExpanded,
		5,
		15,
	)

	require.NoError(t, err)

	labels, children, numCluster, err := target.Fit(
		features,
		featureRow,
		featureCol,
	)

	require.NoError(t, err)

	require.Equal(t, numCluster, int32(5))
	require.Equal(t, len(labels), featureRow)
	require.Equal(t, len(children), (featureRow-1)*2)

}
