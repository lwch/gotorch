package model

import (
	"fmt"
	"testing"
)

func TestLoad(t *testing.T) {
	m, err := Load("yolo_tiny.pt", nil)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("params count: %d\n", len(m.Params()))
}
