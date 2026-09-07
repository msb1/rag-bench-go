package service

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"sync"

	"rag-bench-go/internal/config"
	"rag-bench-go/internal/domain"
	"rag-bench-go/internal/repository"
)

type BatchResult struct {
	Processed int    `json:"processed"`
	Skipped   int    `json:"skipped"`
	Output    string `json:"output"`
}
type Benchmark struct {
	Config    config.Config
	RAG       *RAG
	Evaluator *Evaluator
	mu        sync.Mutex
}
type Query struct {
	Query  string `json:"query"`
	Type   string `json:"type"`
	Source string `json:"source"`
}

func readJSON(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}
func recordHash(row domain.Record) string {
	b, _ := json.Marshal(row)
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
func (b *Benchmark) Run(ctx context.Context, limit int, progress Progress) (BatchResult, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	out := BatchResult{Output: b.Config.DatasetFile}
	queries := map[string]Query{}
	answers := map[string]string{}
	if err := readJSON(b.Config.QueriesFile, &queries); err != nil {
		return out, err
	}
	if err := readJSON(b.Config.AnswersFile, &answers); err != nil {
		return out, err
	}
	completed := map[string]domain.Record{}
	err := repository.ReadJSONL(b.Config.DatasetFile, func(r domain.Record) error {
		if r.QueryID != "" {
			completed[r.QueryID] = r
		}
		return nil
	})
	if err != nil && !os.IsNotExist(err) {
		return out, err
	}
	ids := make([]string, 0, len(queries))
	for id, q := range queries {
		if q.Query == "" {
			return out, fmt.Errorf("empty query %s", id)
		}
		if _, ok := answers[id]; !ok {
			return out, fmt.Errorf("missing ground truth for %s", id)
		}
		ids = append(ids, id)
	}
	sort.Strings(ids)
	pending := []string{}
	for _, id := range ids {
		r, ok := completed[id]
		if ok && r.Question == queries[id].Query && r.GroundTruth == answers[id] {
			out.Skipped++
			continue
		}
		if ok {
			return out, fmt.Errorf("benchmark input changed for %s; select a new DATASET_FILE", id)
		}
		pending = append(pending, id)
	}
	if limit > 0 && len(pending) > limit {
		pending = pending[:limit]
	}
	for i, id := range pending {
		if err = ctx.Err(); err != nil {
			return out, err
		}
		if progress != nil {
			progress(i, len(pending), id)
		}
		result, err := b.RAG.Answer(ctx, queries[id].Query)
		if err != nil {
			return out, fmt.Errorf("query %s: %w", id, err)
		}
		result.QueryID = id
		result.GroundTruth = answers[id]
		if err = repository.AppendJSONL(b.Config.DatasetFile, result); err != nil {
			return out, err
		}
		out.Processed++
		checkpoint, err := json.Marshal(map[string]any{"last_query_id": id, "completed": out.Skipped + out.Processed})
		if err != nil {
			return out, err
		}
		if err = repository.AtomicWrite(b.Config.ProgressFile, checkpoint); err != nil {
			return out, err
		}
		if progress != nil {
			progress(i+1, len(pending), id)
		}
	}
	return out, nil
}
func (b *Benchmark) Evaluate(ctx context.Context, limit int, progress Progress) (BatchResult, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	out := BatchResult{Output: b.Config.OutputFile}
	if b.Config.OutputFile == b.Config.DatasetFile {
		return out, fmt.Errorf("OUTPUT_FILE and DATASET_FILE must differ")
	}
	done := map[string]bool{}
	err := repository.ReadJSONL(b.Config.OutputFile, func(raw json.RawMessage) error {
		r, err := DecodeEvaluated(raw)
		if err != nil {
			return err
		}
		done[recordHash(r.Record)] = true
		return nil
	})
	if err != nil && !os.IsNotExist(err) {
		return out, err
	}
	rows := []domain.Record{}
	err = repository.ReadJSONL(b.Config.DatasetFile, func(r domain.Record) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if done[recordHash(r)] {
			out.Skipped++
			return nil
		}
		if limit == 0 || len(rows) < limit {
			rows = append(rows, r)
		}
		return nil
	})
	if err != nil {
		return out, err
	}
	for i, row := range rows {
		if err = ctx.Err(); err != nil {
			return out, err
		}
		if progress != nil {
			progress(i, len(rows), row.QueryID)
		}
		evaluated, err := b.Evaluator.Evaluate(ctx, row)
		if err != nil {
			return out, err
		}
		if err = repository.AppendJSONL(b.Config.OutputFile, evaluated); err != nil {
			return out, err
		}
		out.Processed++
		if progress != nil {
			progress(i+1, len(rows), row.QueryID)
		}
	}
	return out, nil
}
