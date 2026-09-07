package api

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"rag-bench-go/internal/app"
	"rag-bench-go/internal/domain"
	"rag-bench-go/internal/service"
	"rag-bench-go/resources"
)

type Server struct{ App *app.App }

func New(a *app.App) http.Handler {
	s := &Server{a}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) { write(w, 200, map[string]string{"status": "ok"}) })
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/docs", http.StatusTemporaryRedirect)
	})
	mux.HandleFunc("GET /openapi.json", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(resources.OpenAPI)
	})
	mux.HandleFunc("GET /docs", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write([]byte(resources.Swagger))
	})
	mux.HandleFunc("GET /v1/config", s.config)
	mux.HandleFunc("POST /v1/chunk", s.chunk)
	mux.HandleFunc("POST /v1/index", s.index)
	mux.HandleFunc("POST /v1/rag", s.rag)
	mux.HandleFunc("POST /v1/retrieve", s.retrieve)
	mux.HandleFunc("POST /v1/rerank", s.rerank)
	mux.HandleFunc("POST /v1/evaluate", s.evaluate)
	mux.HandleFunc("POST /v1/evaluate/{metric}", s.metric)
	mux.HandleFunc("POST /v1/benchmark", s.benchmark)
	mux.HandleFunc("POST /v1/eval", s.eval)
	mux.HandleFunc("POST /v1/analyze", s.analyze)
	mux.HandleFunc("GET /v1/analysis", s.analysis)
	mux.HandleFunc("GET /v1/dashboard.svg", s.dashboard)
	mux.HandleFunc("GET /v1/jobs/{id}", s.job)
	mux.HandleFunc("DELETE /v1/jobs/{id}", s.cancel)
	mux.HandleFunc("GET /v1/collection", s.collection)
	mux.HandleFunc("POST /v1/collection", s.ensure)
	mux.HandleFunc("GET /v1/chunks/{id}", s.getChunk)
	mux.HandleFunc("DELETE /v1/documents", s.delete)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		w.Header().Set("X-Content-Type-Options", "nosniff")
		defer func() {
			if p := recover(); p != nil {
				slog.Error("request panic", "path", r.URL.Path, "error", p)
				fail(w, 500, errors.New("internal server error"))
			}
			slog.Info("request", "method", r.Method, "path", r.URL.Path, "duration", time.Since(start))
		}()
		if strings.HasPrefix(r.URL.Path, "/v1/") && a.Config.APIKey != "" {
			token, bearer := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
			if !bearer || subtle.ConstantTimeCompare([]byte(token), []byte(a.Config.APIKey)) != 1 {
				fail(w, 401, errors.New("valid bearer API_KEY required"))
				return
			}
		}
		ctx, cancel := context.WithTimeout(r.Context(), a.Config.RequestTimeout)
		defer cancel()
		mux.ServeHTTP(w, r.WithContext(ctx))
	})
}
func write(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		slog.Error("encode response", "error", err)
	}
}
func fail(w http.ResponseWriter, status int, err error) {
	if errors.Is(err, context.DeadlineExceeded) {
		status = 504
	}
	write(w, status, map[string]string{"error": err.Error()})
}
func read(w http.ResponseWriter, r *http.Request, v any) bool {
	r.Body = http.MaxBytesReader(w, r.Body, 32<<20)
	defer r.Body.Close()
	d := json.NewDecoder(r.Body)
	var raw json.RawMessage
	if err := d.Decode(&raw); err != nil {
		fail(w, 400, fmt.Errorf("invalid JSON body: %w", err))
		return false
	}
	if len(raw) == 0 || bytes.TrimSpace(raw)[0] != '{' {
		fail(w, 400, errors.New("body must be a JSON object"))
		return false
	}
	object := json.NewDecoder(bytes.NewReader(raw))
	object.DisallowUnknownFields()
	if err := object.Decode(v); err != nil {
		fail(w, 400, fmt.Errorf("invalid JSON body: %w", err))
		return false
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		fail(w, 400, errors.New("body must contain exactly one JSON value"))
		return false
	}
	return true
}
func (s *Server) config(w http.ResponseWriter, r *http.Request) {
	c := s.App.Config
	write(w, 200, map[string]any{"collection": c.Collection, "embedding_model": c.Embedding.Name, "rag_model": c.RAG.Name, "judge_model": c.Judge.Name, "fusion_model": c.Fusion.Name, "reranker_model": c.RerankModel, "chunk_tokens": c.ChunkTokens, "chunk_overlap": c.ChunkOverlap, "tokenizer": "cl100k_base", "retrieval_k": c.RetrievalK, "rerank_top_n": c.TopN, "query_variations": c.QueryVariations})
}
func (s *Server) chunk(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Markdown string `json:"markdown"`
		Key      string `json:"key"`
	}
	if !read(w, r, &in) {
		return
	}
	if strings.TrimSpace(in.Markdown) == "" || in.Key == "" {
		fail(w, 400, errors.New("markdown and key are required"))
		return
	}
	chunks, err := s.App.Chunker.Chunk(in.Markdown, in.Key)
	if err != nil {
		fail(w, 400, err)
		return
	}
	write(w, 200, map[string]any{"chunks": chunks, "count": len(chunks)})
}

type limitInput struct {
	Limit int `json:"limit"`
}

func validLimit(w http.ResponseWriter, n int) bool {
	if n < 0 {
		fail(w, 400, errors.New("limit must be nonnegative"))
		return false
	}
	return true
}
func (s *Server) start(w http.ResponseWriter, kind string, run func(context.Context, service.Progress) (any, error)) {
	j, err := s.App.Jobs.Start(kind, run)
	if err != nil {
		status := 500
		if errors.Is(err, service.ErrBusy) {
			status = 409
		}
		fail(w, status, err)
		return
	}
	w.Header().Set("Location", "/v1/jobs/"+j.ID)
	write(w, 202, j)
}
func (s *Server) index(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Keys  []string `json:"keys"`
		Limit int      `json:"limit"`
	}
	if !read(w, r, &in) || !validLimit(w, in.Limit) {
		return
	}
	for _, key := range in.Keys {
		if !s.App.Indexer.Allowed(key) {
			fail(w, 400, errors.New("keys must be Markdown objects under the configured prefix"))
			return
		}
	}
	s.start(w, "index", func(ctx context.Context, p service.Progress) (any, error) {
		return s.App.Indexer.Run(ctx, in.Keys, in.Limit, p)
	})
}

type questionInput struct {
	Question string `json:"question"`
}

func question(w http.ResponseWriter, r *http.Request) (string, bool) {
	var in questionInput
	if !read(w, r, &in) {
		return "", false
	}
	if strings.TrimSpace(in.Question) == "" || len(in.Question) > 32000 {
		fail(w, 400, errors.New("question must contain 1–32000 bytes"))
		return "", false
	}
	return in.Question, true
}
func (s *Server) rag(w http.ResponseWriter, r *http.Request) {
	q, ok := question(w, r)
	if !ok {
		return
	}
	out, err := s.App.RAG.Answer(r.Context(), q)
	if err != nil {
		fail(w, 502, err)
		return
	}
	write(w, 200, out)
}
func (s *Server) retrieve(w http.ResponseWriter, r *http.Request) {
	q, ok := question(w, r)
	if !ok {
		return
	}
	out, err := s.App.RAG.Retrieve(r.Context(), q)
	if err != nil {
		fail(w, 502, err)
		return
	}
	write(w, 200, out)
}
func (s *Server) rerank(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Query     string   `json:"query"`
		Documents []string `json:"documents"`
		TopN      int      `json:"top_n"`
	}
	if !read(w, r, &in) {
		return
	}
	if in.TopN == 0 {
		in.TopN = s.App.Config.TopN
	}
	if strings.TrimSpace(in.Query) == "" || len(in.Documents) > 1000 || in.TopN < 1 || in.TopN > 1000 {
		fail(w, 400, errors.New("query, up to 1000 documents and top_n from 1 to 1000 required"))
		return
	}
	out, err := s.App.Reranker.Rerank(r.Context(), in.Query, in.Documents, in.TopN)
	if err != nil {
		fail(w, 502, err)
		return
	}
	write(w, 200, map[string]any{"results": out})
}
func record(w http.ResponseWriter, r *http.Request) (domain.Record, bool) {
	var row domain.Record
	if !read(w, r, &row) {
		return row, false
	}
	if strings.TrimSpace(row.Question) == "" || row.Contexts == nil || strings.TrimSpace(row.Answer) == "" || strings.TrimSpace(row.GroundTruth) == "" {
		fail(w, 400, errors.New("question, answer, contexts array and ground_truth are required"))
		return row, false
	}
	return row, true
}
func (s *Server) evaluate(w http.ResponseWriter, r *http.Request) {
	row, ok := record(w, r)
	if !ok {
		return
	}
	out, err := s.App.Evaluator.Evaluate(r.Context(), row)
	if err != nil {
		fail(w, 502, err)
		return
	}
	write(w, 200, out)
}
func (s *Server) metric(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("metric")
	found := false
	for _, n := range service.MetricNames {
		if n == name {
			found = true
		}
	}
	if !found {
		fail(w, 404, errors.New("unknown metric"))
		return
	}
	row, ok := record(w, r)
	if !ok {
		return
	}
	out, err := s.App.Evaluator.Metric(r.Context(), name, row)
	if err != nil {
		fail(w, 502, err)
		return
	}
	write(w, 200, map[string]any{"metric": name, "score": out})
}
func (s *Server) benchmark(w http.ResponseWriter, r *http.Request) {
	var in limitInput
	if !read(w, r, &in) || !validLimit(w, in.Limit) {
		return
	}
	s.start(w, "benchmark", func(ctx context.Context, p service.Progress) (any, error) {
		return s.App.Benchmark.Run(ctx, in.Limit, p)
	})
}
func (s *Server) eval(w http.ResponseWriter, r *http.Request) {
	var in limitInput
	if !read(w, r, &in) || !validLimit(w, in.Limit) {
		return
	}
	s.start(w, "eval", func(ctx context.Context, p service.Progress) (any, error) {
		return s.App.Benchmark.Evaluate(ctx, in.Limit, p)
	})
}
func (s *Server) analyze(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Rows   []json.RawMessage `json:"rows"`
		Export bool              `json:"export"`
	}
	if !read(w, r, &in) {
		return
	}
	var out service.Analysis
	var err error
	if in.Rows == nil {
		out, err = service.AnalyzeFile(s.App.Config)
	} else {
		rows := make([]domain.Evaluated, 0, len(in.Rows))
		for _, raw := range in.Rows {
			row, decodeErr := service.DecodeEvaluated(raw)
			if decodeErr != nil {
				fail(w, 400, decodeErr)
				return
			}
			rows = append(rows, row)
		}
		out, err = service.Analyze(rows, .3, .3)
	}
	if err != nil {
		fail(w, 400, err)
		return
	}
	if in.Export {
		err = service.ExportAnalysis(s.App.Config, out)
		if err != nil {
			fail(w, 500, err)
			return
		}
	}
	write(w, 200, out)
}
func (s *Server) analysis(w http.ResponseWriter, r *http.Request) {
	out, err := service.AnalyzeFile(s.App.Config)
	if err != nil {
		fail(w, 400, err)
		return
	}
	write(w, 200, out)
}
func (s *Server) dashboard(w http.ResponseWriter, r *http.Request) {
	out, err := service.AnalyzeFile(s.App.Config)
	if err != nil {
		fail(w, 400, err)
		return
	}
	w.Header().Set("Content-Type", "image/svg+xml")
	_, _ = w.Write([]byte(service.Dashboard(out)))
}
func (s *Server) job(w http.ResponseWriter, r *http.Request) {
	j, ok := s.App.Jobs.Get(r.PathValue("id"))
	if !ok {
		fail(w, 404, errors.New("job not found"))
		return
	}
	write(w, 200, j)
}
func (s *Server) cancel(w http.ResponseWriter, r *http.Request) {
	if !s.App.Jobs.Cancel(r.PathValue("id")) {
		fail(w, 404, errors.New("job not found"))
		return
	}
	j, _ := s.App.Jobs.Get(r.PathValue("id"))
	write(w, 202, j)
}
func (s *Server) collection(w http.ResponseWriter, r *http.Request) {
	info, err := s.App.Store.Info(r.Context())
	if err != nil {
		fail(w, 502, err)
		return
	}
	write(w, 200, map[string]any{"collection": s.App.Config.Collection, "points_count": info.GetPointsCount(), "status": info.GetStatus().String(), "config": info.GetConfig()})
}
func (s *Server) ensure(w http.ResponseWriter, r *http.Request) {
	vectors, err := s.App.Embedding.Embed(r.Context(), []string{"embedding dimension probe"})
	if err == nil {
		err = s.App.Store.Ensure(r.Context(), len(vectors[0]))
	}
	if err != nil {
		fail(w, 502, err)
		return
	}
	write(w, 200, map[string]string{"collection": s.App.Config.Collection, "status": "ready"})
}
func (s *Server) getChunk(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if _, err := uuid.Parse(id); err != nil {
		fail(w, 400, errors.New("chunk ID must be a UUID"))
		return
	}
	out, err := s.App.Store.Get(r.Context(), id)
	if err != nil {
		fail(w, 502, err)
		return
	}
	if len(out) == 0 {
		fail(w, 404, errors.New("chunk not found"))
		return
	}
	write(w, 200, out[0])
}
func (s *Server) delete(w http.ResponseWriter, r *http.Request) {
	key := r.URL.Query().Get("key")
	if !s.App.Indexer.Allowed(key) {
		fail(w, 400, errors.New("key must be a Markdown object under the configured prefix"))
		return
	}
	if err := s.App.Indexer.Delete(r.Context(), key); err != nil {
		fail(w, 502, err)
		return
	}
	write(w, 200, map[string]string{"deleted_source": key})
}
