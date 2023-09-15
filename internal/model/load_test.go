package model

import (
	"fmt"
	"testing"
)

func TestLoad(t *testing.T) {
	m, err := Load("./test/linear.pt")
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println(m.params["linear.linear"].Get())
}
