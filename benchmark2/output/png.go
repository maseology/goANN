package output

import (
	"image/color"
	"log"
	"math"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func ToPng(fp string, o, s []float64) {
	sequentialLine := func(v []float64) plotter.XYs {
		pts, c := make(plotter.XYs, len(v)), 0
		for i := range pts {
			if math.IsNaN(v[i]) {
				continue
			}
			pts[c].X = float64(i)
			pts[c].Y = v[i]
			c++
		}
		return pts[:c]
	}

	p := plot.New()

	// p.Title.Text = fp
	p.X.Label.Text = ""
	p.Y.Label.Text = "discharge"

	ps, err := plotter.NewLine(sequentialLine(s))
	if err != nil {
		log.Fatalf(" obsSim error: %v", err)
	}
	ps.Color = color.RGBA{R: 255, A: 255}

	po, err := plotter.NewLine(sequentialLine(o))
	if err != nil {
		log.Fatalf(" obsSim error: %v", err)
	}
	po.Color = color.RGBA{B: 255, A: 255}

	// Add the functions and their legend entries.
	p.Add(ps, po)
	p.Legend.Add("obs", po)
	p.Legend.Add("sim", ps)
	p.Legend.Top = true
	// p.X.Tick.Marker = plot.TimeTicks{Format: "Jan"}

	// Save the plot to a PNG file.
	if err := p.Save(24*vg.Inch, 8*vg.Inch, fp); err != nil {
		log.Fatalf(" obsSim error: %v", err)
	}
}
