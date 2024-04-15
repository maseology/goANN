package goann

import "math/rand"

// NewNet m: number of input nodes; n: nodes per hidden layer; nhl: number of hidden layers; p: number of output nodes
func NewNet(m, n, p, nhl int, eta float64) Network {
	init := func() float64 { return .25 * (2.*rand.Float64() - 1.) }

	if nhl > 1 { // "deep-learning"
		nn := n * nhl
		nodes := make([]*node, m+nn+p)
		for i := 0; i < m; i++ {
			nodes[i] = &node{b: nil, f: make([]*weight, n)}
		}
		for l := 0; l < nhl; l++ {
			ll := l * n
			if l == 0 { // first hidden layer
				for j := 0; j < n; j++ {
					nodes[m+j] = &node{b: make([]*weight, m), f: make([]*weight, n)}
				}
			} else if l == nhl-1 { // last hidden layer
				for j := 0; j < n; j++ {
					nodes[ll+m+j] = &node{b: make([]*weight, n), f: make([]*weight, p)}
				}
			} else {
				for j := 0; j < n; j++ {
					nodes[ll+m+j] = &node{b: make([]*weight, n), f: make([]*weight, n)}
				}
			}
		}
		for k := 0; k < p; k++ {
			nodes[m+nn+k] = &node{b: make([]*weight, n), f: nil}
		}

		// connectivity (only single hidden layer for now)
		for i := 0; i < m; i++ { // connection into hidden layer
			for j := 0; j < n; j++ {
				w := weight{w: init(), b: nodes[i], f: nodes[m+j]}
				nodes[i].f[j] = &w
				nodes[m+j].b[i] = &w
			}
		}

		for l := 0; l < nhl-1; l++ { // hidden layers
			llm := l*n + m
			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					w := weight{w: init(), b: nodes[llm+i], f: nodes[llm+n+j]}
					nodes[llm+i].f[j] = &w
					nodes[llm+n+j].b[i] = &w
				}
			}
		}

		llm := (nhl-1)*n + m
		for j := 0; j < n; j++ { // outputs from hidden layer
			for k := 0; k < p; k++ {
				w := weight{w: init(), b: nodes[llm+j], f: nodes[llm+n+k]}
				nodes[llm+j].f[k] = &w
				nodes[llm+n+k].b[j] = &w
			}
		}

		return Network{
			nd:  nodes,
			eta: eta,
			m:   m,
			n:   n,
			p:   p,
		}

	}

	// only a single hidden layer (original/old code)
	nodes := make([]*node, m+n+p)
	for i := 0; i < m; i++ {
		nodes[i] = &node{b: nil, f: make([]*weight, n)}
	}
	for j := 0; j < n; j++ {
		nodes[m+j] = &node{b: make([]*weight, m), f: make([]*weight, p)}
	}
	for k := 0; k < p; k++ {
		nodes[m+n+k] = &node{b: make([]*weight, n), f: nil}
	}

	// connectivity (only single hidden layer for now)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			w := weight{w: init(), b: nodes[i], f: nodes[m+j]}
			nodes[i].f[j] = &w
			nodes[m+j].b[i] = &w
		}
	}
	for j := 0; j < n; j++ {
		for k := 0; k < p; k++ {
			w := weight{w: init(), b: nodes[m+j], f: nodes[m+n+k]}
			nodes[m+j].f[k] = &w
			nodes[m+n+k].b[j] = &w
		}
	}

	return Network{
		nd:  nodes,
		eta: eta,
		m:   m,
		n:   n,
		p:   p,
	}
}
