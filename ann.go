package goann

import (
	"math/rand"
	"time"
)

const eta = .5

type ANN struct {
	w          [][][]float64 // weights
	h, b       [][]float64   // hidden layers; biases
	l, m, n, p int           // number of inputs, hidden neurons per layer, outputs
}

// based on Zhu M-L., M. Fujita and N. Hashimoto. 1994. Application of Neural Networks to Runoff Prediction in Time Series Analysis in Hydrology and Environmental Engineering ed. K.W. Hippel, A.I. McLeod, U.S. Panu and V.P. Singh. Water Science adn Technology Library. 474pp.
func (a *ANN) Initialize(m, p, n, depth int) {
	a.w = make([][][]float64, depth+1) // weights
	a.h = make([][]float64, depth)     // hidden neurons ("perceptrons")
	a.b = make([][]float64, depth+1)   // biases
	a.m = m                            // number of input neurons
	a.p = p                            // number of output neurons (special case: many input to singular output)
	a.n = n                            // number of hidden neurons/perceptrons per layer
	a.l = depth                        // number of hidden layers (>1 is "deep learning")

	// initialize
	rand.Seed(time.Now().UnixNano())
	negToOne := func() float64 {
		return 2.*rand.Float64() - 1.
	}

	for l := 0; l < depth; l++ {
		a.h[l] = make([]float64, a.n)
		a.w[l] = make([][]float64, a.n)
		a.b[l] = make([]float64, a.n)
		for j := 0; j < a.n; j++ {
			a.w[l][j] = make([]float64, a.m)
			a.b[l][j] = negToOne()
			for i := 0; i < a.m; i++ {
				a.w[l][j][i] = negToOne()
			}
		}

	}
	a.w[depth] = make([][]float64, a.p)
	a.b[depth] = make([]float64, a.p)
	for k := 0; k < p; k++ {
		a.w[depth][k] = make([]float64, a.n)
		for j := 0; j < a.n; j++ {
			a.w[depth][k][j] = negToOne()
		}
		a.b[depth][k] = negToOne()
	}

}

func (a *ANN) Train(input, trainer []float64) ([]float64, float64) {
	// input
	h0 := make([]float64, a.n) // first line of hidden neurons
	for j := 0; j < a.n; j++ {
		for i, oi := range input {
			h0[j] += oi*a.w[0][j][i] + a.b[0][j]
		}
		a.h[0][j] = sigmoid(h0[j])
	}

	// "deep" layers
	if len(a.h) > 1 {
		panic("todo")
	}

	// output
	o, output := make([]float64, a.p), make([]float64, a.p)
	for k := 0; k < a.p; k++ {
		for j := 0; j < a.n; j++ {
			o[k] += a.h[a.l-1][j]*a.w[a.l][k][j] + a.b[a.l][k]
		}
		output[k] = o[k] //sigmoid(o[k])
	}

	delta := 0.
	// back-propagation
	deltaOutput := make([]float64, a.p)
	deltaHidden := make([][]float64, a.l)
	for k := 0; k < a.p; k++ {
		deltaOutput[k] = (trainer[k] - output[k]) * sigmoidPrime(o[k]) // error
		delta += deltaOutput[k]
		deltaHidden[a.l-1] = make([]float64, a.n)
		for j := 0; j < a.n; j++ {
			a.w[a.l][k][j] += eta * deltaOutput[k] * a.h[a.l-1][j]
			deltaHidden[a.l-1][j] += deltaOutput[k] * a.w[a.l][k][j] * sigmoidPrime(h0[j])
		}
		a.b[a.l][k] += eta * deltaOutput[k]

		// "deep" layers
		if len(a.h) > 1 {
			panic("todo")
		}

		// input layer
		for j := 0; j < a.n; j++ {
			for i, oi := range input {
				a.w[0][j][i] += eta * deltaHidden[0][j] * oi
			}
			a.b[0][j] += eta * deltaHidden[0][j]
		}
	}

	return output, delta
}
