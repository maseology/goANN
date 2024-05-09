package goann

import "math"

func tanhPrime(x float64) float64 {
	t := math.Tanh(x)
	return 1. - t*t
}

func sigmoid(x float64) float64 {
	return 1. / (1. + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	s := sigmoid(x)
	return s * (1. - s)
}

func relu(x float64) float64 {
	if x < 0. {
		return 0.
	}
	return x
}

func reluPrime(x float64) float64 {
	if x < 0. {
		return 0.
	}
	return 1.
}
