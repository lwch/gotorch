//go:build linux && !gpu
// +build linux,!gpu

package torch

// #cgo CXXFLAGS: -std=c++17
// #cgo LDFLAGS: -ltorch -lc10 -ltorch_cpu
import "C"
