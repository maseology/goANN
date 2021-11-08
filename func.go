package goann

import "math"

func sigmoid(x float64) float64 {
	return 1. / (1. + math.Exp(-x))
}

// derivative of the sigmoid function at x
func sigmoidPrime(x float64) float64 {
	s := sigmoid(x)
	return s * (1. - s)
}
