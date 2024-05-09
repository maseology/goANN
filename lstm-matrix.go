package goann

// LTSM matrix formulation modified from https://github.com/nicodjimenez/lstm

import (
	"math"
	"math/rand"
)

// randArr create uniform random array w/ values in [a,b) and shape args
func randArr(a, b float64, nr, nc int) [][]float64 {
	o := make([][]float64, nr)
	for i := 0; i < nr; i++ {
		o[i] = make([]float64, nc)
		for j := 0; j < nc; j++ {
			o[i][j] = rand.Float64()*(b-a) + a
		}
	}
	return o
}

func zeros(nr, nc int) [][]float64 {
	o := make([][]float64, nr)
	for i := 0; i < nr; i++ {
		o[i] = make([]float64, nc)
	}
	return o
}

func dot(a, b []float64) float64 {
	s := 0.
	for i, aa := range a {
		s += aa * b[i]
	}
	return s
}

func hstack(a, b []float64) []float64 {
	s := make([]float64, len(a)+len(b))
	for i, aa := range a {
		s[i] = aa
	}
	for i, bb := range b {
		s[i+len(a)] = bb
	}
	return s
}

func transpose(a [][]float64) [][]float64 {
	o := make([][]float64, len(a[0]))
	for j := 0; j < len(a[0]); j++ {
		o[j] = make([]float64, len(a))
		for i := 0; i < len(a); i++ {
			o[j][i] = a[i][j]
		}
	}
	return o
}

func outer(a, b []float64) [][]float64 {
	o := make([][]float64, len(a))
	for i, aa := range a {
		o[i] = make([]float64, len(b))
		for j, bb := range b {
			o[i][j] = aa * bb
		}
	}
	return o
}

// LTSMparam modified from https://github.com/nicodjimenez/lstm
type LTSMparam struct {
	wg, wi, wf, wo, wgDiff, wiDiff, wfDiff, woDiff [][]float64
	bg, bi, bf, bo, bgDiff, biDiff, bfDiff, boDiff []float64
	mem_cell_ct, x_dim                             int
}

func NewLTSMparam(mem_cell_ct, x_dim int) LTSMparam {
	concat_len := x_dim + mem_cell_ct
	return LTSMparam{
		// weight matrices
		wg: randArr(-0.1, 0.1, mem_cell_ct, concat_len),
		wi: randArr(-0.1, 0.1, mem_cell_ct, concat_len),
		wf: randArr(-0.1, 0.1, mem_cell_ct, concat_len),
		wo: randArr(-0.1, 0.1, mem_cell_ct, concat_len),
		// bias terms
		bg: randArr(-0.1, 0.1, 1, mem_cell_ct)[0],
		bi: randArr(-0.1, 0.1, 1, mem_cell_ct)[0],
		bf: randArr(-0.1, 0.1, 1, mem_cell_ct)[0],
		bo: randArr(-0.1, 0.1, 1, mem_cell_ct)[0],
		// diffs (derivative of loss function w.r.t. all parameters)
		wgDiff: zeros(mem_cell_ct, concat_len),
		wiDiff: zeros(mem_cell_ct, concat_len),
		wfDiff: zeros(mem_cell_ct, concat_len),
		woDiff: zeros(mem_cell_ct, concat_len),
		bgDiff: zeros(1, mem_cell_ct)[0],
		biDiff: zeros(1, mem_cell_ct)[0],
		bfDiff: zeros(1, mem_cell_ct)[0],
		boDiff: zeros(1, mem_cell_ct)[0],

		mem_cell_ct: mem_cell_ct,
		x_dim:       x_dim,
	}
}

func (l *LTSMparam) ApplyDiff(lr float64) {
	concat_len := l.x_dim + l.mem_cell_ct
	for i := 0; i < l.mem_cell_ct; i++ {
		for j := 0; j < concat_len; j++ {
			l.wg[i][j] -= lr * l.wgDiff[i][j]
			l.wi[i][j] -= lr * l.wiDiff[i][j]
			l.wf[i][j] -= lr * l.wfDiff[i][j]
			l.wo[i][j] -= lr * l.woDiff[i][j]
		}
		l.bg[i] -= lr * l.bgDiff[i]
		l.bi[i] -= lr * l.biDiff[i]
		l.bf[i] -= lr * l.bfDiff[i]
		l.bo[i] -= lr * l.boDiff[i]
	}

	// reset diffs to zero
	l.wgDiff = zeros(l.mem_cell_ct, concat_len)
	l.wiDiff = zeros(l.mem_cell_ct, concat_len)
	l.wfDiff = zeros(l.mem_cell_ct, concat_len)
	l.woDiff = zeros(l.mem_cell_ct, concat_len)
	l.bgDiff = zeros(1, l.mem_cell_ct)[0]
	l.biDiff = zeros(1, l.mem_cell_ct)[0]
	l.bfDiff = zeros(1, l.mem_cell_ct)[0]
	l.boDiff = zeros(1, l.mem_cell_ct)[0]
}

type LSTMstate struct{ g, i, f, o, s, H, bottomDiffh, bottomDiffs []float64 }

func NewLTSMstate(mem_cell_ct int) LSTMstate {
	return LSTMstate{
		g:           zeros(1, mem_cell_ct)[0],
		i:           zeros(1, mem_cell_ct)[0],
		f:           zeros(1, mem_cell_ct)[0],
		o:           zeros(1, mem_cell_ct)[0],
		s:           zeros(1, mem_cell_ct)[0],
		H:           zeros(1, mem_cell_ct)[0],
		bottomDiffh: zeros(1, mem_cell_ct)[0],
		bottomDiffs: zeros(1, mem_cell_ct)[0],
	}
}

type LSTMnode struct {
	State            LSTMstate
	param            LTSMparam
	xc, sPrev, hPrev []float64 // xc: non-recurrent input concatenated with recurrent input
}

func NewLSTMnode(ls LSTMstate, lp LTSMparam) LSTMnode {
	return LSTMnode{State: ls, param: lp}
}

func (ln *LSTMnode) bottomDataIs(x, sPrev, hPrev []float64) {
	// if this is the first lstm node in the network
	if sPrev == nil {
		sPrev = zeros(1, ln.param.mem_cell_ct)[0]
	}
	if hPrev == nil {
		hPrev = zeros(1, ln.param.mem_cell_ct)[0]
	}
	// save data for use in backprop
	ln.sPrev = sPrev
	ln.hPrev = hPrev

	ln.xc = hstack(x, hPrev) // concatenate x(t) and h(t-1)
	for i := 0; i < ln.param.mem_cell_ct; i++ {
		ln.State.g[i] = math.Tanh(dot(ln.param.wg[i], ln.xc) + ln.param.bg[i])
		ln.State.i[i] = sigmoid(dot(ln.param.wi[i], ln.xc) + ln.param.bi[i])
		ln.State.f[i] = sigmoid(dot(ln.param.wf[i], ln.xc) + ln.param.bf[i])
		ln.State.o[i] = sigmoid(dot(ln.param.wo[i], ln.xc) + ln.param.bo[i])
		ln.State.s[i] = ln.State.g[i]*ln.State.i[i] + sPrev[i]*ln.State.f[i]
		ln.State.H[i] = ln.State.s[i] * ln.State.o[i]
	}
}

func (ln *LSTMnode) topDiffIs(topDiffh, topDiffs []float64) {
	concat_len := ln.param.x_dim + ln.param.mem_cell_ct
	// notice that top_diff_s is carried along the constant error carousel
	ds, doInput, diInput, dgInput, dfInput := make([]float64, ln.param.mem_cell_ct), make([]float64, ln.param.mem_cell_ct), make([]float64, ln.param.mem_cell_ct), make([]float64, ln.param.mem_cell_ct), make([]float64, ln.param.mem_cell_ct)
	for i := 0; i < ln.param.mem_cell_ct; i++ {
		ds[i] = ln.State.o[i]*topDiffh[i] + topDiffs[i]
		do := ln.State.s[i] * topDiffh[i]
		di := ln.State.g[i] * ds[i]
		dg := ln.State.i[i] * ds[i]
		df := ln.sPrev[i] * ds[i]

		// diffs w.r.t. vector inside sigma / tanh function
		diInput[i] = sigmoidPrime(ln.State.i[i]) * di
		dfInput[i] = sigmoidPrime(ln.State.f[i]) * df
		doInput[i] = sigmoidPrime(ln.State.o[i]) * do
		dgInput[i] = tanhPrime(ln.State.g[i]) * dg

		diO, dfO, doO, dgO := outer(diInput, ln.xc), outer(dfInput, ln.xc), outer(doInput, ln.xc), outer(dgInput, ln.xc)
		for j := 0; j < concat_len; j++ {
			// diffs w.r.t. inputs
			ln.param.wiDiff[i][j] += diO[i][j]
			ln.param.wfDiff[i][j] += dfO[i][j]
			ln.param.woDiff[i][j] += doO[i][j]
			ln.param.wgDiff[i][j] += dgO[i][j]
		}
		ln.param.biDiff[i] += diInput[i]
		ln.param.bfDiff[i] += dfInput[i]
		ln.param.boDiff[i] += doInput[i]
		ln.param.bgDiff[i] += dgInput[i]
	}

	// compute bottom diff
	dxc := zeros(1, concat_len)[0]
	wiT, wfT, woT, wgT := transpose(ln.param.wi), transpose(ln.param.wf), transpose(ln.param.wo), transpose(ln.param.wg)
	for i := 0; i < concat_len; i++ {
		dxc[i] += dot(wiT[i], diInput)
		dxc[i] += dot(wfT[i], dfInput)
		dxc[i] += dot(woT[i], doInput)
		dxc[i] += dot(wgT[i], dgInput)
	}

	// save bottom diffs
	for i := 0; i < ln.param.mem_cell_ct; i++ {
		ln.State.bottomDiffs[i] = ds[i] * ln.State.f[i]
		ln.State.bottomDiffh[i] = dxc[i+ln.param.x_dim]
	}
}

type LSTMnetwork struct {
	param    LTSMparam
	NodeList []LSTMnode
	xList    [][]float64 // input sequence
}

func NewLSTMnetwork(lp LTSMparam) LSTMnetwork {
	return LSTMnetwork{param: lp, NodeList: []LSTMnode{}, xList: [][]float64{}}
}

func (lw *LSTMnetwork) YListIs(yList []float64) float64 {
	/*
	   Updates diffs by setting target sequence
	   with corresponding loss layer.
	   Will *NOT* update parameters. To update parameters,
	   call self.lstm_param.apply_diff()
	*/
	if len(yList) != len(lw.xList) {
		panic("lw.yListIs ERROR 1")
	}
	idx := len(lw.xList) - 1
	// first node only gets diffs from label ...
	lossLayer := func(pred []float64, label float64) float64 {
		f := pred[0] * label
		return f * f // Computes square loss with first element of hidden layer array.
	}
	bottomDiffLayer := func(pred []float64, label float64) []float64 {
		o := make([]float64, len(pred))
		o[0] = 2 * pred[0] * label
		return o
	}
	loss := lossLayer(lw.NodeList[idx].State.H, yList[idx])
	diffh := bottomDiffLayer(lw.NodeList[idx].State.H, yList[idx])
	// here s is not affecting loss due to h(t+1), hence we set equal to zero
	diffs := zeros(1, lw.param.mem_cell_ct)[0]
	lw.NodeList[idx].topDiffIs(diffh, diffs)
	idx--

	// ... following nodes also get diffs from next nodes, hence we add diffs to diffh
	// we also propagate error along constant error carousel using diffs
	for idx >= 0 {
		loss += lossLayer(lw.NodeList[idx].State.H, yList[idx])
		diffh = bottomDiffLayer(lw.NodeList[idx].State.H, yList[idx])
		for i := 0; i < lw.param.mem_cell_ct; i++ {
			diffh[i] += lw.NodeList[idx+1].State.bottomDiffh[i]
		}
		diffs = lw.NodeList[idx+1].State.bottomDiffs
		lw.NodeList[idx].topDiffIs(diffh, diffs)
		idx--
	}
	return loss
}

func (lw *LSTMnetwork) XlistClear() {
	lw.xList = make([][]float64, lw.param.x_dim)
}

func (lw *LSTMnetwork) XlistAdd(x []float64) {
	lw.xList = append(lw.xList, x)
	if len(lw.xList) > len(lw.NodeList) {
		// need to add new lstm node, create new state mem
		ls := NewLTSMstate(lw.param.mem_cell_ct)
		lw.NodeList = append(lw.NodeList, NewLSTMnode(ls, lw.param))
	}

	// get index of most recent x input
	idx := len(lw.xList) - 1
	if idx == 0 {
		// no recurrent inputs yet
		lw.NodeList[idx].bottomDataIs(x, nil, nil)
	} else {
		sPrev := lw.NodeList[idx-1].State.s
		hPrev := lw.NodeList[idx-1].State.H
		lw.NodeList[idx].bottomDataIs(x, sPrev, hPrev)
	}
}

/*
self.x_list.append(x)
if len(self.x_list) > len(self.lstm_node_list):
	# need to add new lstm node, create new state mem
	lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
	self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))

# get index of most recent x input
idx = len(self.x_list) - 1
if idx == 0:
	# no recurrent inputs yet
	self.lstm_node_list[idx].bottom_data_is(x)
else:
	s_prev = self.lstm_node_list[idx - 1].state.s
	h_prev = self.lstm_node_list[idx - 1].state.h
	self.lstm_node_list[idx].bottom_data_is(x, s_prev, h_prev)


/*
class LstmParam:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct
        # weight matrices
        self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        # bias terms
        self.bg = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bi = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bf = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bo = rand_arr(-0.1, 0.1, mem_cell_ct)
        # diffs (derivative of loss function w.r.t. all parameters)
        self.wg_diff = np.zeros((mem_cell_ct, concat_len))
        self.wi_diff = np.zeros((mem_cell_ct, concat_len))
        self.wf_diff = np.zeros((mem_cell_ct, concat_len))
        self.wo_diff = np.zeros((mem_cell_ct, concat_len))
        self.bg_diff = np.zeros(mem_cell_ct)
        self.bi_diff = np.zeros(mem_cell_ct)
        self.bf_diff = np.zeros(mem_cell_ct)
        self.bo_diff = np.zeros(mem_cell_ct)

    def apply_diff(self, lr = 1):
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff
        # reset diffs to zero
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi)
        self.wf_diff = np.zeros_like(self.wf)
        self.wo_diff = np.zeros_like(self.wo)
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi)
        self.bf_diff = np.zeros_like(self.bf)
        self.bo_diff = np.zeros_like(self.bo)

class LstmState:
    def __init__(self, mem_cell_ct, x_dim):
        self.g = np.zeros(mem_cell_ct)
        self.i = np.zeros(mem_cell_ct)
        self.f = np.zeros(mem_cell_ct)
        self.o = np.zeros(mem_cell_ct)
        self.s = np.zeros(mem_cell_ct)
        self.h = np.zeros(mem_cell_ct)
        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)

class LstmNode:
    def __init__(self, lstm_param, lstm_state):
        # store reference to parameters and to activations
        self.state = lstm_state
        self.param = lstm_param
        # non-recurrent input concatenated with recurrent input
        self.xc = None

    def bottom_data_is(self, x, s_prev = None, h_prev = None):
        # if this is the first lstm node in the network
        if s_prev is None: s_prev = np.zeros_like(self.state.s)
        if h_prev is None: h_prev = np.zeros_like(self.state.h)
        # save data for use in backprop
        self.s_prev = s_prev
        self.h_prev = h_prev

        # concatenate x(t) and h(t-1)
        xc = np.hstack((x,  h_prev))
        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.h = self.state.s * self.state.o

        self.xc = xc

    def top_diff_is(self, top_diff_h, top_diff_s):
        # notice that top_diff_s is carried along the constant error carousel
        ds = self.state.o * top_diff_h + top_diff_s
        do = self.state.s * top_diff_h
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev * ds

        # diffs w.r.t. vector inside sigma / tanh function
        di_input = sigmoid_derivative(self.state.i) * di
        df_input = sigmoid_derivative(self.state.f) * df
        do_input = sigmoid_derivative(self.state.o) * do
        dg_input = tanh_derivative(self.state.g) * dg

        # diffs w.r.t. inputs
        self.param.wi_diff += np.outer(di_input, self.xc)
        self.param.wf_diff += np.outer(df_input, self.xc)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input

        # compute bottom diff
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        # save bottom diffs
        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_h = dxc[self.param.x_dim:]

class LstmNetwork():
    def __init__(self, lstm_param):
        self.lstm_param = lstm_param
        self.lstm_node_list = []
        # input sequence
        self.x_list = []

    def y_list_is(self, y_list, loss_layer):
        """
        Updates diffs by setting target sequence
        with corresponding loss layer.
        Will *NOT* update parameters.  To update parameters,
        call self.lstm_param.apply_diff()
        """
        assert len(y_list) == len(self.x_list)
        idx = len(self.x_list) - 1
        # first node only gets diffs from label ...
        loss = loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
        diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
        # here s is not affecting loss due to h(t+1), hence we set equal to zero
        diff_s = np.zeros(self.lstm_param.mem_cell_ct)
        self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
        idx -= 1

        ### ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
        ### we also propagate error along constant error carousel using diff_s
        while idx >= 0:
            loss += loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
            diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
            diff_h += self.lstm_node_list[idx + 1].state.bottom_diff_h
            diff_s = self.lstm_node_list[idx + 1].state.bottom_diff_s
            self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
            idx -= 1

        return loss

    def x_list_clear(self):
        self.x_list = []

    def x_list_add(self, x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.lstm_node_list):
            # need to add new lstm node, create new state mem
            lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
            self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))

        # get index of most recent x input
        idx = len(self.x_list) - 1
        if idx == 0:
            # no recurrent inputs yet
            self.lstm_node_list[idx].bottom_data_is(x)
        else:
            s_prev = self.lstm_node_list[idx - 1].state.s
            h_prev = self.lstm_node_list[idx - 1].state.h
            self.lstm_node_list[idx].bottom_data_is(x, s_prev, h_prev)


*/
