package main

import (
	"fmt"
	"math/rand"
	"time"
	"zTest/benchmark2/dset"
	"zTest/benchmark2/output"

	"github.com/maseology/goHydro/pet"
	"github.com/maseology/objfunc"
)

func main() {
	fmt.Println("Training..")
	t1 := time.Now()

	nhn := 3
	tlag := 3

	net := CreateNetwork(tlag*2+1, nhn, 1, 0.1)
	owrcTrain(&net, "../02EC018.csv", tlag)

	elapsed := time.Since(t1)
	fmt.Printf("Time taken to train: %s\n nhn: %d  tlag: %d", elapsed, nhn, tlag)
}

func owrcTrain(net *Network, fp string, tlag int) {
	rand.Seed(time.Now().UTC().UnixNano())

	ts, dat := dset.ReadOWRC(fp)
	ts = ts[tlag+1:]

	input, qTrain := func() (o [][]float64, q []float64) {
		// collect
		yield, q := make([]float64, len(dat)), make([]float64, len(dat))
		for i, v := range dat {
			yield[i] = v.Yeild()
			q[i] = v.Runoff()
		}

		// re-scale
		yield = objfunc.RescaleLim(yield, .1, .85)
		q = objfunc.RescaleLim(q, .1, .85)

		o = make([][]float64, len(ts))
		for i, t := range ts {
			o[i] = make([]float64, net.inputs)
			o[i][net.inputs-1] = pet.SineCurve(t)
			for ii := 0; ii < tlag; ii++ {
				o[i][ii] = yield[i+tlag-ii]
			}
		}
		for i := 0; i < tlag; i++ {
			o[0][tlag+i] = q[tlag-i-1]
		}
		q = q[tlag+1:]
		return
	}()

	for epochs := 0; epochs < 2.5e7/len(ts); epochs++ {
		for i := tlag; i < len(ts)-1; i++ {
			net.Train(input[i], []float64{qTrain[i]})
			for j := 0; j < tlag; j++ {
				input[i+1][tlag+j] = net.Predict(input[i-j-1]).At(0, 0) // "partial recurrent neural network."
			}
		}
	}

	func() { // print
		obs, sim := make([]float64, len(ts)), make([]float64, len(ts))
		for i := range ts {
			sim[i] = (net.Predict(input[i]).At(0, 0) - .1) / (.85 - .1)
			obs[i] = (qTrain[i] - .1) / (.85 - .1)
		}
		fmt.Println(objfunc.NSE(obs, sim))
		output.ToPng("hyd.png", obs, sim)
		output.ToCsv("hyd.csv", ts, obs, sim)
	}()
}
