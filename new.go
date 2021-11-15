package goann

import (
	"math/rand"
	"time"
)

func NewNet(m, n, p int, eta float64) Network {
	rand.Seed(time.Now().UTC().UnixNano())
	init := func() float64 { return .25 * (2.*rand.Float64() - 1.) }

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
