package output

import (
	"log"
	"time"

	"github.com/maseology/mmio"
)

func ToCsv(fp string, ts []time.Time, obs, sim []float64) {
	csvw := mmio.NewCSVwriter(fp)
	defer csvw.Close()
	if err := csvw.WriteHead("date,obs,sim"); err != nil {
		log.Fatalf("%v", err)
	}
	for i, y := range ts {
		csvw.WriteLine(y, obs[i], sim[i])
	}
}
