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
	for i := 0; i < 1000; i++ {
		fmt.Println(m.params[fmt.Sprintf("linear.%d.linear", i)].Get())
	}
}
