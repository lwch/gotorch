//go:build windows && !gpu
// +build windows,!gpu

package torch

// #cgo LDFLAGS: -lgotorch
import "C"
