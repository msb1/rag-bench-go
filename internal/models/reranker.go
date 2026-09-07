package models

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
)

type Rank struct {
	Index int     `json:"index"`
	Score float64 `json:"relevance_score"`
}
type Reranker interface {
	Rerank(context.Context, string, []string, int) ([]Rank, error)
}
type Jina struct {
	Endpoint, Model, Key string
	HTTP                 *http.Client
}

func (j *Jina) Rerank(ctx context.Context, query string, docs []string, n int) ([]Rank, error) {
	if len(docs) == 0 {
		return []Rank{}, nil
	}
	body, err := json.Marshal(map[string]any{"model": j.Model, "query": query, "documents": docs, "top_n": n, "return_documents": false})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, j.Endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if j.Key != "" {
		req.Header.Set("Authorization", "Bearer "+j.Key)
	}
	resp, err := j.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("reranker: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("reranker HTTP %d", resp.StatusCode)
	}
	var payload struct {
		Results []struct {
			Index *int     `json:"index"`
			Score *float64 `json:"relevance_score"`
		} `json:"results"`
	}
	if err = json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&payload); err != nil {
		return nil, fmt.Errorf("reranker response: %w", err)
	}
	if len(payload.Results) == 0 {
		return nil, fmt.Errorf("reranker returned no results")
	}
	out := []Rank{}
	seen := map[int]bool{}
	for _, r := range payload.Results {
		if r.Index == nil || r.Score == nil || *r.Index < 0 || *r.Index >= len(docs) || seen[*r.Index] || math.IsNaN(*r.Score) || math.IsInf(*r.Score, 0) {
			return nil, fmt.Errorf("invalid reranker result")
		}
		seen[*r.Index] = true
		out = append(out, Rank{*r.Index, *r.Score})
	}
	sort.SliceStable(out, func(i, k int) bool { return out[i].Score > out[k].Score })
	return out[:min(n, len(out))], nil
}
