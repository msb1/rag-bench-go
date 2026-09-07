package service

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"

	"rag-bench-go/internal/chunker"
	"rag-bench-go/internal/models"
	"rag-bench-go/internal/repository"
)

type Progress func(done, total int, detail string)
type IndexResult struct {
	Documents int `json:"documents"`
	Chunks    int `json:"chunks"`
	Skipped   int `json:"skipped"`
}
type Indexer struct {
	Documents repository.Documents
	Store     repository.VectorStore
	Embedding models.Embedder
	Chunker   *chunker.Chunker
	BatchSize int
	Prefix    string
	mu        sync.Mutex
}

func (s *Indexer) Run(ctx context.Context, keys []string, limit int, progress Progress) (IndexResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	result := IndexResult{}
	var err error
	if len(keys) == 0 {
		keys, err = s.Documents.List(ctx, s.Prefix)
		if err != nil {
			return result, err
		}
	}
	sort.Strings(keys)
	keys = compact(keys)
	if limit > 0 && len(keys) > limit {
		keys = keys[:limit]
	}
	for _, key := range keys {
		if !s.Allowed(key) {
			return result, fmt.Errorf("key must be a Markdown object under the configured S3 prefix")
		}
	}
	initialized := false
	for i, key := range keys {
		if err = ctx.Err(); err != nil {
			return result, err
		}
		if progress != nil {
			progress(i, len(keys), key)
		}
		markdown, err := s.Documents.Read(ctx, key)
		if err != nil {
			return result, fmt.Errorf("%s: %w", key, err)
		}
		chunks, err := s.Chunker.Chunk(markdown, key)
		if err != nil {
			return result, err
		}
		if len(chunks) == 0 {
			result.Skipped++
			continue
		} // An empty/noise document never erases a previous index.
		ids := make([]string, len(chunks))
		for start := 0; start < len(chunks); start += s.BatchSize {
			batch := chunks[start:min(start+s.BatchSize, len(chunks))]
			texts := make([]string, len(batch))
			for j, c := range batch {
				texts[j] = c.Text
				if strings.TrimSpace(texts[j]) == "" {
					texts[j] = "Table data"
				}
				ids[start+j] = c.ID
			}
			embeddings, err := s.Embedding.Embed(ctx, texts)
			if err != nil {
				return result, err
			}
			if !initialized {
				if err = s.Store.Ensure(ctx, len(embeddings[0])); err != nil {
					return result, err
				}
				initialized = true
			}
			if err = s.Store.Upsert(ctx, batch, embeddings); err != nil {
				return result, fmt.Errorf("upsert %s: %w", key, err)
			}
		}
		if err = s.Store.Prune(ctx, key, ids); err != nil {
			return result, fmt.Errorf("prune stale chunks for %s: %w", key, err)
		}
		result.Documents++
		result.Chunks += len(chunks)
		if progress != nil {
			progress(i+1, len(keys), key)
		}
	}
	return result, nil
}
func compact(keys []string) []string {
	out := []string{}
	for _, k := range keys {
		if len(out) == 0 || out[len(out)-1] != k {
			out = append(out, k)
		}
	}
	return out
}
func (s *Indexer) Allowed(key string) bool {
	p := strings.Trim(s.Prefix, "/")
	lower := strings.ToLower(key)
	return key != "" && (p == "" || strings.HasPrefix(key, p+"/")) && (strings.HasSuffix(lower, ".md") || strings.HasSuffix(lower, ".markdown"))
}
func (s *Indexer) Delete(ctx context.Context, key string) error {
	if !s.Allowed(key) {
		return fmt.Errorf("key must be a Markdown object under the configured S3 prefix")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.Store.Prune(ctx, key, nil)
}
