package main

import (
	"fmt"
	"math/rand"
	"time"
	"zTest/benchmark1/mnist"
)

// modified from: https://sausheong.github.io/posts/how-to-build-a-simple-artificial-neural-network-with-go/
func main() {
	// 784 inputs - 28 x 28 pixels, each pixel is an input
	// 200 hidden neurons - an arbitrary number
	// 10 outputs - digits 0 to 9
	// 0.1 is the learning rate
	net := CreateNetwork(784, 200, 10, 0.1)

	// mnist := flag.String("mnist", "", "Either train or predict to evaluate neural network")
	// flag.Parse()

	// // train or mass predict to determine the effectiveness of the trained network
	// switch *mnist {
	// case "train":
	mnistTrain(&net)
	save(net)
	// case "predict":
	load(&net)
	mnistPredict(&net)
	// default:
	// 	// don't do anything
	// }
}

func mnistTrain(net *Network) {
	fmt.Println("Training..")
	rand.Seed(time.Now().UTC().UnixNano())
	t1 := time.Now()

	ls := mnist.GetLbl("../dat/train-labels-idx1-ubyte.gz", 60000, false)   // .gz files at: http://yann.lecun.com/exdb/mnist/
	imgs := mnist.GetImg("../dat/train-images-idx3-ubyte.gz", 60000, false) // .gz files at: http://yann.lecun.com/exdb/mnist/

	for epochs := 0; epochs < 5; epochs++ {
		for j, a := range imgs {

			inputs := make([]float64, net.inputs)
			for i := range inputs {
				inputs[i] = (float64(a[i]) / 255.0 * 0.99) + 0.01
			}

			targets := make([]float64, 10)
			for i := range targets {
				targets[i] = 0.01
			}
			targets[int(ls[j])] = 0.99

			net.Train(inputs, targets)
		}
	}
	elapsed := time.Since(t1)
	fmt.Printf("\nTime taken to train: %s\n", elapsed)
}

func mnistPredict(net *Network) {
	fmt.Println("Predicting..")
	t1 := time.Now()

	ls := mnist.GetLbl("../dat/t10k-labels-idx1-ubyte.gz", 10000, false)   // .gz files at: http://yann.lecun.com/exdb/mnist/
	imgs := mnist.GetImg("../dat/t10k-images-idx3-ubyte.gz", 10000, false) // .gz files at: http://yann.lecun.com/exdb/mnist/

	score := 0
	for j, a := range imgs {

		inputs := make([]float64, net.inputs)
		for i := range inputs {
			inputs[i] = (float64(a[i]) / 255.0 * 0.99) + 0.01
		}

		outputs := net.Predict(inputs)
		best := 0
		highest := 0.0
		for i := 0; i < net.outputs; i++ {
			if outputs.At(i, 0) > highest {
				best = i
				highest = outputs.At(i, 0)
			}
		}

		if best == int(ls[j]) {
			score++
		}
	}

	elapsed := time.Since(t1)
	fmt.Printf("Time taken to check: %s\n", elapsed)
	fmt.Println("score:", score)
}
