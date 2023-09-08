package cuml4go_test

import (
	"testing"

	cuml4go "github.com/getumen/cuml/go"
	"github.com/stretchr/testify/require"
)

func TestDBScan(t *testing.T) {

	features := csvToFloat32Array(t, "../testdata/feature.csv")
	featureCol := 30
	featureRow := 114

	target := cuml4go.NewDBScan(
		5,
		3.0,
		cuml4go.L2SqrtUnexpanded,
		0,
		cuml4go.Info,
	)

	result, err := target.Fit(
		features,
		featureRow,
		featureCol,
	)

	require.NoError(t, err)

	require.Equal(t, len(result), featureRow)
}
