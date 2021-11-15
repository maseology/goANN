package mnist

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"image"
	"image/png"
	"log"
	"os"
)

// functions to aquire training and test images from: http://yann.lecun.com/exdb/mnist/
func GetImg(gzfp string, n int, prnt bool) [][]byte {

	fmt.Printf("openning %s..\n", gzfp)
	b, err := os.ReadFile(gzfp)
	if err != nil {
		log.Fatalf(" getImg fail-1: %v\n", err)
	}

	// buf := bytes.NewReader(b) // if file already decompressed, OR:

	var buf bytes.Buffer
	gzbuf := bytes.NewBuffer(b)
	r, err := gzip.NewReader(gzbuf)
	if err != nil {
		log.Fatalf(" getImg fail-2: %v\n", err)
	}
	_, err = buf.ReadFrom(r)
	if err != nil {
		log.Fatalf(" getImg fail-3: %v\n", err)
	}

	// skip header
	head := make([]byte, 16)
	buf.Read(head)

	o := make([][]byte, n)
	for i := 0; i < n; i++ {
		pixels := make([]byte, 28*28)
		for j := 0; j < 28*28; j++ {
			pixels[j], _ = buf.ReadByte()
			pixels[j] = 255 - pixels[j]
		}
		o[i] = pixels

		if prnt {
			img := image.NewGray(image.Rect(0, 0, 28, 28))
			copy(img.Pix, pixels)

			out, err := os.Create(fmt.Sprintf("img%d.png", i))
			if err != nil {
				log.Fatalf(" getImg print fail-4: %v\n", err)
			}
			if err := png.Encode(out, img); err != nil {
				log.Fatalf(" getImg print fail-5 3: %v\n", err)
			}
		}
	}
	return o
}

func GetLbl(gzfp string, n int, prnt bool) []byte {

	fmt.Printf("openning %s..\n", gzfp)
	b, err := os.ReadFile(gzfp) // .gz files at: http://yann.lecun.com/exdb/mnist/
	if err != nil {
		log.Fatalf(" getLbl fail-1: %v\n", err)
	}

	// buf := bytes.NewReader(b) // if file already decompressed, OR:

	var buf bytes.Buffer
	gzbuf := bytes.NewBuffer(b)
	r, err := gzip.NewReader(gzbuf)
	if err != nil {
		log.Fatalf(" getImg fail-2: %v\n", err)
	}
	_, err = buf.ReadFrom(r)
	if err != nil {
		log.Fatalf(" getImg fail-3: %v\n", err)
	}

	// skip header
	head := make([]byte, 8)
	buf.Read(head)

	o := make([]byte, n)
	for i := 0; i < n; i++ {
		if o[i], err = buf.ReadByte(); err != nil {
			log.Fatalf(" getLbl fail-4: %v\n", err)
		}
	}
	return o
}
