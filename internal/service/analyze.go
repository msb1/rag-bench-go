package service

import (
	"bytes"
	"encoding/json"
	"fmt"
	"html"
	"math"
	"sort"
	"strings"

	"rag-bench-go/internal/config"
	"rag-bench-go/internal/domain"
	"rag-bench-go/internal/repository"
)

type Statistics struct {
	Count  int     `json:"count"`
	Mean   float64 `json:"mean"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	Std    float64 `json:"std"`
	Q1     float64 `json:"q1"`
	Median float64 `json:"median"`
	Q3     float64 `json:"q3"`
}
type Analysis struct {
	Rows            int                   `json:"rows"`
	Metrics         map[string]Statistics `json:"metrics"`
	Hallucinations  []domain.Evaluated    `json:"hallucinations"`
	RetrievalMisses []domain.Evaluated    `json:"retrieval_misses"`
}

func quantile(v []float64, p float64) float64 {
	pos := float64(len(v)-1) * p
	lo := int(pos)
	hi := min(lo+1, len(v)-1)
	return v[lo] + (v[hi]-v[lo])*(pos-float64(lo))
}
func Analyze(rows []domain.Evaluated, faithThreshold, recallThreshold float64) (Analysis, error) {
	out := Analysis{Rows: len(rows), Metrics: map[string]Statistics{}, Hallucinations: []domain.Evaluated{}, RetrievalMisses: []domain.Evaluated{}}
	if faithThreshold < 0 || faithThreshold > 1 || recallThreshold < 0 || recallThreshold > 1 {
		return out, fmt.Errorf("thresholds must be between 0 and 1")
	}
	values := map[string][]float64{}
	for _, r := range rows {
		for k, v := range r.Metrics.Values() {
			if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 || v > 1 {
				return out, fmt.Errorf("invalid %s score", k)
			}
			values[k] = append(values[k], v)
		}
		if r.Faithfulness < faithThreshold {
			out.Hallucinations = append(out.Hallucinations, r)
		}
		if r.ContextRecall < recallThreshold {
			out.RetrievalMisses = append(out.RetrievalMisses, r)
		}
	}
	for k, v := range values {
		sort.Float64s(v)
		s := Statistics{Count: len(v), Min: v[0], Max: v[len(v)-1], Q1: quantile(v, .25), Median: quantile(v, .5), Q3: quantile(v, .75)}
		for _, n := range v {
			s.Mean += n
		}
		s.Mean /= float64(len(v))
		if len(v) > 1 {
			for _, n := range v {
				s.Std += (n - s.Mean) * (n - s.Mean)
			}
			s.Std = math.Sqrt(s.Std / float64(len(v)-1))
		}
		out.Metrics[k] = s
	}
	return out, nil
}
func AnalyzeFile(c config.Config) (Analysis, error) {
	rows := []domain.Evaluated{}
	err := repository.ReadJSONL(c.OutputFile, func(r json.RawMessage) error {
		row, err := DecodeEvaluated(r)
		if err != nil {
			return err
		}
		rows = append(rows, row)
		return nil
	})
	if err != nil {
		return Analysis{}, err
	}
	return Analyze(rows, .3, .3)
}

func DecodeEvaluated(data []byte) (domain.Evaluated, error) {
	var row domain.Evaluated
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return row, err
	}
	for _, name := range MetricNames {
		if len(fields[name]) == 0 || string(fields[name]) == "null" {
			return row, fmt.Errorf("missing metric %s", name)
		}
	}
	if err := json.Unmarshal(data, &row); err != nil {
		return row, err
	}
	for name, v := range row.Metrics.Values() {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 || v > 1 {
			return row, fmt.Errorf("invalid %s score", name)
		}
	}
	return row, nil
}
func ExportAnalysis(c config.Config, a Analysis) error {
	if err := repository.WriteJSONL(c.HallucinationsFile, a.Hallucinations); err != nil {
		return err
	}
	if err := repository.WriteJSONL(c.MissesFile, a.RetrievalMisses); err != nil {
		return err
	}
	return repository.AtomicWrite(c.DashboardFile, []byte(Dashboard(a)))
}

// Dashboard is a self-contained SVG artifact: mean bars and distribution boxes.
func Dashboard(a Analysis) string {
	var b bytes.Buffer
	fmt.Fprintf(&b, `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="440" viewBox="0 0 1200 440" role="img" aria-label="RAG evaluation dashboard"><rect width="1200" height="440" fill="#f8fafc"/><g font-family="system-ui,sans-serif" fill="#0f172a"><text x="32" y="42" font-size="24">RAG benchmark — %d evaluated answers</text><text x="240" y="90" font-size="16">Mean scores</text><text x="810" y="90" font-size="16">Distribution: min, quartiles, max</text>`, a.Rows)
	for i, name := range MetricNames {
		s := a.Metrics[name]
		y := 130 + i*48
		label := html.EscapeString(strings.ReplaceAll(name, "_", " "))
		fmt.Fprintf(&b, `<text x="32" y="%d" font-size="15">%s</text><rect x="210" y="%d" width="320" height="24" rx="4" fill="#e2e8f0"/><rect x="210" y="%d" width="%.2f" height="24" rx="4" fill="#0d9488"/><text x="540" y="%d" font-size="14">%.3f</text>`, y+17, label, y, y, 320*s.Mean, y+17, s.Mean)
		x := func(v float64) float64 { return 790 + v*330 }
		fmt.Fprintf(&b, `<line x1="%.2f" y1="%d" x2="%.2f" y2="%d" stroke="#64748b" stroke-width="2"/><rect x="%.2f" y="%d" width="%.2f" height="24" fill="#c4b5fd" stroke="#7c3aed"/><line x1="%.2f" y1="%d" x2="%.2f" y2="%d" stroke="#5b21b6" stroke-width="3"/>`, x(s.Min), y+12, x(s.Max), y+12, x(s.Q1), y, max(1.0, x(s.Q3)-x(s.Q1)), x(s.Median), y, x(s.Median), y+24)
	}
	fmt.Fprintf(&b, `<text x="32" y="410" font-size="14">Severe hallucinations (&lt;0.30): %d · Retrieval misses (&lt;0.30): %d · Score range: 0–1</text></g></svg>`, len(a.Hallucinations), len(a.RetrievalMisses))
	return b.String()
}
