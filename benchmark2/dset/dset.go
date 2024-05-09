package dset

import (
	"io"
	"log"
	"os"
	"strconv"
	"time"

	"github.com/maseology/mmio"
)

type dset struct{ q, tx, tn, rf, sf, sm, pa float64 }

func (d *dset) Yeild() float64 { return d.rf + d.sm }

func (d *dset) Runoff() float64 { return d.q }

func ReadOWRC(csvfp string) ([]time.Time, []dset) {

	f, err := os.Open(csvfp)
	if err != nil {
		log.Fatalf("readOWRC failed: %v\n", err)
	}
	defer f.Close()

	recs := mmio.LoadCSV(io.Reader(f), 1) // "Date","Flow","Flag","Tx","Tn","Rf","Sf","Sm","Pa"
	o, ts := make([]dset, 0), make([]time.Time, 0)
	for rec := range recs {
		// fmt.Println(rec)
		t, err := time.Parse("2006-01-02", rec[0])
		if err != nil {
			log.Fatalf("readOWRC date read fail: %v\n", err)
		}
		// fmt.Println(t)
		g := func(i int) float64 {
			v, err := strconv.ParseFloat(rec[i], 64)
			if err != nil {
				if rec[i] == "NA" {
					return 0. //math.NaN()
				}
				log.Fatalf("readOWRC date read fail: value parse error: %v (%d)", err, i)
			}
			return v
		}

		ts = append(ts, t)
		o = append(o, dset{
			q:  g(1),
			tx: g(3),
			tn: g(4),
			rf: g(5),
			sf: g(6),
			sm: g(7),
			pa: g(8),
		})
	}

	return ts, o
}
