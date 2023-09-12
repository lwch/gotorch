package pickle

import (
	"os"
	"testing"
)

func TestDecode(t *testing.T) {
	f, err := os.Open("data.pkl")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	dec, err := NewDecoder(f)
	if err != nil {
		t.Fatal(err)
	}
	var ret interface{}
	err = dec.Unmarshal(&ret)
	if err != nil {
		t.Fatal(err)
	}
}
