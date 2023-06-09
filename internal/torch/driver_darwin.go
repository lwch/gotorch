//go:build darwin
// +build darwin

package torch

// #cgo CXXFLAGS: -std=c++17
// #cgo LDFLAGS: -ltorch -lc10 -ltorch_cpu
import "C"
