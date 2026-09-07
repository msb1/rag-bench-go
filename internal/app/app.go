package app

import (
	"context"
	"net/http"

	"rag-bench-go/internal/chunker"
	"rag-bench-go/internal/config"
	"rag-bench-go/internal/models"
	"rag-bench-go/internal/repository"
	"rag-bench-go/internal/service"
)

type App struct {
	Config    config.Config
	Chunker   *chunker.Chunker
	S3        *repository.S3
	Store     *repository.Qdrant
	Embedding *models.OpenAI
	Reranker  *models.Jina
	RAG       *service.RAG
	Evaluator *service.Evaluator
	Indexer   *service.Indexer
	Benchmark *service.Benchmark
	Jobs      *service.Jobs
	http      *http.Client
}

func New(ctx context.Context, c config.Config) (*App, error) {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.MaxIdleConnsPerHost = 16
	h := &http.Client{Timeout: c.HTTPTimeout, Transport: transport}
	ch, err := chunker.New(c.ChunkTokens, c.ChunkOverlap)
	if err != nil {
		return nil, err
	}
	s3, err := repository.NewS3(ctx, c, h)
	if err != nil {
		return nil, err
	}
	store, err := repository.NewQdrant(c)
	if err != nil {
		return nil, err
	}
	embedding := models.NewOpenAI(c.Embedding, h)
	rerank := &models.Jina{Endpoint: c.RerankEndpoint, Model: c.RerankModel, Key: c.RerankKey, HTTP: h}
	rag := &service.RAG{Embedding: embedding, LLM: models.NewOpenAI(c.RAG, h), Fusion: models.NewOpenAI(c.Fusion, h), Reranker: rerank, Store: store, Variations: c.QueryVariations, K: c.RetrievalK, TopN: c.TopN}
	evaluator := &service.Evaluator{Judge: models.NewOpenAI(c.Judge, h)}
	return &App{Config: c, Chunker: ch, S3: s3, Store: store, Embedding: embedding, Reranker: rerank, RAG: rag, Evaluator: evaluator, Indexer: &service.Indexer{Documents: s3, Store: store, Embedding: embedding, Chunker: ch, BatchSize: c.BatchSize, Prefix: c.S3Prefix}, Benchmark: &service.Benchmark{Config: c, RAG: rag, Evaluator: evaluator}, Jobs: service.NewJobs(ctx), http: h}, nil
}
func (a *App) Close() error { a.http.CloseIdleConnections(); return a.Store.Close() }
