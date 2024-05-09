package goann

import "math"

// LSTM follows the nomenclature of Kratzert et.al., 2018
type LSTM struct{ Wf, Wg, Wi, Wo, Uf, Ug, Ui, Uo, Bf, Bg, Bi, Bo, h, c float64 }

type LSTMlayers struct {
	layer []LSTM
	eta   float64
	nl    int
}

func (ls *LSTMlayers) reset() {
	for i := 0; i < ls.nl; i++ {
		// ls.layer[i].Wf = 0.
		// ls.layer[i].Wg = 0.
		// ls.layer[i].Wi = 0.
		// ls.layer[i].Wo = 0.
		// ls.layer[i].Uf = 0.
		// ls.layer[i].Ug = 0.
		// ls.layer[i].Ui = 0.
		// ls.layer[i].Uo = 0.
		// ls.layer[i].Bf = 0.
		// ls.layer[i].Bg = 0.
		// ls.layer[i].Bi = 0.
		// ls.layer[i].Bo = 0.
		ls.layer[i].h = 0.
		ls.layer[i].c = 0.
	}
}

func (l *LSTM) update(x float64) (g, i, f, o float64) {
	g = l.Wg*x + l.Ug*l.h + l.Bg                   // candidate state
	i = l.Wi*x + l.Ui*l.h + l.Bi                   // input gate
	f = l.Wf*x + l.Uf*l.h + l.Bf                   // forget gate
	o = l.Wo*x + l.Uo*l.h + l.Bo                   // output gate
	l.c = sigmoid(f)*l.c + sigmoid(i)*math.Tanh(g) // update cell state
	l.h = math.Tanh(l.c) * sigmoid(o)              // update hidden state
	return
}

func (l *LSTM) backpropagate(e, x, g, i, f, o, c0, h0 float64) {
	// Gradient with respect to output gate weights:
	do := e * math.Tanh(l.c) * sigmoidPrime(o)
	l.Bo += do
	l.Wo += do * x
	l.Uo += do * h0

	// Gradient with respect to forget gate weights:
	dc := e * o * tanhPrime(l.c)
	df := dc * c0 * sigmoidPrime(f)
	l.Bf += df
	l.Wf += df * x
	l.Uf += df * h0

	// Gradient with respect to input gate weights:
	di := dc * g * sigmoidPrime(i)
	l.Bi += di
	l.Wi += di * x
	l.Ui += di * h0
	dg := dc * i * tanhPrime(g)
	l.Bg += dg
	l.Wg += dg * x
	l.Ug += dg * h0
}

func (ls *LSTMlayers) Train(input, trainer []float64) {
	// forward propagate
	ls.reset()

	// save predicted timeseries
	ypred := make([]float64, len(trainer))

	// saving recursive states for back-propagation
	g, i, f, o := make([][]float64, ls.nl), make([][]float64, ls.nl), make([][]float64, ls.nl), make([][]float64, ls.nl)
	clast, hlast := make([][]float64, ls.nl), make([][]float64, ls.nl)
	for k := 1; k < ls.nl; k++ { // initialize
		g[k], i[k], f[k], o[k] = make([]float64, len(trainer)), make([]float64, len(trainer)), make([]float64, len(trainer)), make([]float64, len(trainer))
		clast[k], hlast[k] = make([]float64, len(trainer)), make([]float64, len(trainer))
	}

	for j, v := range input {
		clast[0][j], hlast[0][j] = ls.layer[0].c, ls.layer[0].h
		g[0][j], i[0][j], f[0][j], o[0][j] = ls.layer[0].update(v)

		// deep learning
		for k := 1; k < ls.nl; k++ {
			clast[k][j], hlast[k][j] = ls.layer[k].c, ls.layer[k].h
			g[k][j], i[k][j], f[k][j], o[k][j] = ls.layer[k].update(ls.layer[k-1].h)
		}

		// prediction
		ypred[j] = ls.layer[ls.nl-1].h
	}

	// back propagate errors
	for j, y := range trainer {
		e := y - ypred[j] // mse
		for k := 0; k < ls.nl; k++ {
			ls.layer[k].backpropagate(ls.eta*e, input[j], g[k][j], i[k][j], f[k][j], o[k][j], clast[k][j], hlast[k][j])
		}
	}
}
