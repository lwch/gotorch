//go:build linux && !gpu
// +build linux,!gpu

package tensor

// #cgo CXXFLAGS: -std=c++17
// #cgo LDFLAGS: -ltorch -lc10 -ltorch_cpu
import "C"
