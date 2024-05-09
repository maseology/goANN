package goann

// NewNet nl: number of recurrent layers; eta learning rate
func NewLSTM(nl int, eta float64) LSTMlayers {
	return LSTMlayers{
		layer: make([]LSTM, nl),
		eta:   eta,
		nl:    nl,
	}
}
