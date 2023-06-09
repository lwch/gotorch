//go:build windows && !gpu
// +build windows,!gpu

package torch

// #cgo CXXFLAGS: -std=c++17
// #cgo LDFLAGS: -ltorch -lc10 -ltorch_cpu
import "C"
