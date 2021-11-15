package goann

type node struct {
	b, f       []*weight
	h, e, bias float64
}

type weight struct {
	b, f *node
	w    float64
}

type Network struct {
	nd      []*node
	eta     float64
	m, n, p int
}

func (nn *Network) reset() {
	for i := 0; i < nn.m+nn.n+nn.p; i++ {
		nn.nd[i].h = 0.
		nn.nd[i].e = 0.
	}
}

func (nn *Network) Feed(input []float64) []float64 {
	// forward propagate
	nn.reset()
	for i := 0; i < nn.m; i++ {
		nn.nd[i].h = input[i]
		for _, w := range nn.nd[i].f {
			w.f.h += w.w * nn.nd[i].h
		}
	}
	for j := 0; j < nn.n; j++ {
		jj := nn.m + j
		for _, w := range nn.nd[jj].f {
			w.f.h += w.w * sigmoid(nn.nd[jj].h+nn.nd[jj].bias)
		}
	}
	o := make([]float64, nn.p)
	for k := 0; k < nn.p; k++ {
		kk := nn.m + nn.n + k
		o[k] = sigmoid(nn.nd[kk].h + nn.nd[kk].bias)
	}
	return o
}

func (nn *Network) Train(input, trainer []float64) {
	// forward propagate
	nn.reset()
	for i := 0; i < nn.m; i++ {
		nn.nd[i].h = input[i]
		for _, w := range nn.nd[i].f {
			w.f.h += w.w * nn.nd[i].h
		}
	}
	for j := 0; j < nn.n; j++ {
		jj := nn.m + j
		for _, w := range nn.nd[jj].f {
			w.f.h += w.w * sigmoid(nn.nd[jj].h+nn.nd[jj].bias)
		}
	}

	// back propagate errors
	for k := 0; k < nn.p; k++ {
		n := nn.nd[nn.m+nn.n+k]
		y := sigmoid(n.h + n.bias)
		yp := sigmoidPrime(n.h + n.bias)
		e := (trainer[k] - y)
		for _, w := range n.b {
			w.b.e += w.w * e
			w.w += nn.eta * e * yp * sigmoid(w.b.h+w.b.bias)
		}
		n.bias += nn.eta * e * yp
	}

	for j := nn.n - 1; j >= 0; j-- {
		n := nn.nd[nn.m+j]
		yp := sigmoidPrime(n.h + n.bias)
		for _, w := range n.b {
			w.b.e += w.w * n.e
			w.w += nn.eta * n.e * yp * w.b.h // only for w.b.h = inputs, otherwise, for deep networks, use sigmoid(w.b.h+w.b.bias)
		}
		n.bias += nn.eta * n.e * yp
	}
}

func (nn *Network) TrainNoBias(input, trainer []float64) {
	// forward propagate
	nn.reset()
	for i := 0; i < nn.m; i++ {
		nn.nd[i].h = input[i]
		for _, w := range nn.nd[i].f {
			w.f.h += w.w * nn.nd[i].h
		}
	}
	for j := 0; j < nn.n; j++ {
		jj := nn.m + j
		for _, w := range nn.nd[jj].f {
			w.f.h += w.w * sigmoid(nn.nd[jj].h)
		}
	}

	// back propagate errors
	for k := 0; k < nn.p; k++ {
		n := nn.nd[nn.m+nn.n+k]
		y := sigmoid(n.h)
		yp := sigmoidPrime(n.h)
		e := (trainer[k] - y)
		for _, w := range n.b {
			w.b.e += w.w * e
			w.w += nn.eta * e * yp * sigmoid(w.b.h)
		}
	}

	for j := nn.n - 1; j >= 0; j-- {
		n := nn.nd[nn.m+j]
		yp := sigmoidPrime(n.h)
		for _, w := range n.b {
			w.b.e += w.w * n.e
			w.w += nn.eta * n.e * yp * w.b.h // only for w.b.h = inputs, otherwise, for deep networks, use sigmoid(w.b.h)
		}
	}
}
