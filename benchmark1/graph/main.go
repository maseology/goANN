package main

import (
	"fmt"
	"time"
	"zTest/benchmark1/mnist"

	goann "github.com/maseology/goANN"
)

func main() {
	net := goann.NewNet(784, 200, 10, .1)
	train(&net)
	predict(&net, 784, 10)
}

func train(net *goann.Network) {
	fmt.Println("Training..")
	t1 := time.Now()

	ls := mnist.GetLbl("../dat/train-labels-idx1-ubyte.gz", 60000, false)   // .gz files at: http://yann.lecun.com/exdb/mnist/
	imgs := mnist.GetImg("../dat/train-images-idx3-ubyte.gz", 60000, false) // .gz files at: http://yann.lecun.com/exdb/mnist/

	for epochs := 0; epochs < 5; epochs++ {
		for j, a := range imgs {

			inputs := make([]float64, 784)
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

func predict(net *goann.Network, m, p int) {
	fmt.Println("Predicting..")
	t1 := time.Now()

	ls := mnist.GetLbl("../dat/t10k-labels-idx1-ubyte.gz", 10000, false)   // .gz files at: http://yann.lecun.com/exdb/mnist/
	imgs := mnist.GetImg("../dat/t10k-images-idx3-ubyte.gz", 10000, false) // .gz files at: http://yann.lecun.com/exdb/mnist/

	score := 0
	for j, a := range imgs {

		inputs := make([]float64, m)
		for i := range inputs {
			inputs[i] = (float64(a[i]) / 255.0 * 0.99) + 0.01
		}

		outputs := net.Feed(inputs)
		best := 0
		highest := 0.0
		for i := 0; i < p; i++ {
			if outputs[i] > highest {
				best = i
				highest = outputs[i]
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

// func save(net goann.Network) {
// 	f, err := os.Create("sv/goann.gob")
// 	defer f.Close()
// 	if err != nil {
// 		log.Fatalln(err)
// 	}
// 	enc := gob.NewEncoder(f)
// 	err = enc.Encode(net)
// 	if err != nil {
// 		log.Fatalln(err)
// 	}
// }

// // load a neural network from file
// func load() *goann.Network {
// 	var d goann.Network
// 	f, err := os.Open("sv/goann.gob")
// 	defer f.Close()
// 	if err != nil {
// 		log.Fatalln(err)
// 	}
// 	enc := gob.NewDecoder(f)
// 	err = enc.Decode(&d)
// 	if err != nil {
// 		log.Fatalln(err)
// 	}
// 	return &d
// }
