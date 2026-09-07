// Package sparse implements native lexical BM25 term weights. Qdrant supplies IDF.
// This versioned vocabulary deliberately does not claim FastEmbed compatibility.
package sparse

import (
	"hash/fnv"
	"regexp"
	"sort"
	"strings"
)

const Version = "go-bm25-fnv1a-v1"

type Vector struct {
	Indices []uint32
	Values  []float32
}

var token = regexp.MustCompile(`[\p{L}\p{N}]+`)
var stop = func() map[string]bool {
	m := map[string]bool{}
	for _, s := range strings.Fields("a an and are as at be been being but by for from had has have he her hers him his i in into is it its me my of on or our ours she that the their theirs them they this those to us was we were what when where which who will with you your") {
		m[s] = true
	}
	return m
}()

func encode(text string, query bool) Vector {
	freq := map[uint32]int{}
	length := 0
	for _, s := range token.FindAllString(strings.ToLower(text), -1) {
		if stop[s] {
			continue
		}
		h := fnv.New32a()
		_, _ = h.Write([]byte(s))
		freq[h.Sum32()]++
		length++
	}
	out := Vector{Indices: make([]uint32, 0, len(freq)), Values: make([]float32, 0, len(freq))}
	for id := range freq {
		out.Indices = append(out.Indices, id)
	}
	sort.Slice(out.Indices, func(i, j int) bool { return out.Indices[i] < out.Indices[j] })
	for _, id := range out.Indices {
		v := 1.0
		if !query {
			f := float64(freq[id])
			v = f * 2.2 / (f + 1.2*(0.25+0.75*float64(length)/256.0))
		}
		out.Values = append(out.Values, float32(v))
	}
	return out
}
func Document(text string) Vector { return encode(text, false) }
func Query(text string) Vector    { return encode(text, true) }
