package main

import (
	"fmt"
	"image"
	"image/png"
	"log"
	"os"

	"github.com/maseology/goANN/benchmark1/mnist"
)

func main() {

	imgs := mnist.GetImg("../dat/t10k-images-idx3-ubyte.gz", 10000, false) // .gz files at: http://yann.lecun.com/exdb/mnist/

	for i := 0; i < 10; i++ {
		img := image.NewGray(image.Rect(0, 0, 28, 28))
		copy(img.Pix, imgs[i])

		out, err := os.Create(fmt.Sprintf("img%02d.png", i))
		if err != nil {
			log.Fatalln(err)
		}
		if err := png.Encode(out, img); err != nil {
			log.Fatalln(err)
		}
	}
}
