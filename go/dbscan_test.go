package cuml4go_test

import (
	"testing"

	cuml4go "github.com/getumen/cuml/go"
	"github.com/stretchr/testify/require"
)

func TestDBScan(t *testing.T) {

	features := csvToFloat32Array(t, "testdata/feature.csv")
	featureCol := 30
	featureRow := 114

	result, err := cuml4go.DBScan(
		features,
		featureRow,
		featureCol,
		5,
		3.0,
		0,
		0,
	)

	require.NoError(t, err)

	require.Equal(t, len(result), featureRow)
}
