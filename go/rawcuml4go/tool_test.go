package rawcuml4go_test

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"testing"
)

func csvToFloat32Array(t *testing.T, csvPath string) []float32 {
	data := make([]float32, 0)

	f, err := os.Open(csvPath)
	if err != nil {
		t.Fatal(err)
	}
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		values := strings.Split(scanner.Text(), ",")
		for _, valueString := range values {
			value, err := strconv.ParseFloat(valueString, 32)
			if err != nil {
				t.Fatal(err)
			}
			data = append(data, float32(value))
		}
	}

	return data
}
