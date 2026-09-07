package config

import (
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/joho/godotenv"
)

type Model struct{ Endpoint, Name, Key string }
type Config struct {
	Root, Listen, APIKey                                                                                           string
	S3Endpoint, S3Region, S3Bucket, S3Prefix, S3AccessKey, S3SecretKey                                             string
	QdrantHost, QdrantKey, Collection                                                                              string
	QdrantPort                                                                                                     int
	QdrantTLS                                                                                                      bool
	Embedding, RAG, Judge, Fusion                                                                                  Model
	RerankEndpoint, RerankModel, RerankKey                                                                         string
	ChunkTokens, ChunkOverlap, BatchSize, RetrievalK, TopN, QueryVariations                                        int
	HTTPTimeout, RequestTimeout                                                                                    time.Duration
	MaxDocumentBytes                                                                                               int64
	QueriesFile, AnswersFile, DatasetFile, OutputFile, ProgressFile, HallucinationsFile, MissesFile, DashboardFile string
}

// Load accepts the legacy Python-style .env (including its stray import line).
// Process environment > .env.local > .env > defaults. It never logs values.
func Load(root string) (Config, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return Config{}, err
	}
	values := map[string]string{}
	for _, name := range []string{".env", ".env.local"} {
		data, err := os.ReadFile(filepath.Join(root, name))
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return Config{}, err
		}
		var lines []string
		for _, l := range strings.Split(string(data), "\n") {
			if strings.TrimSpace(l) == "from pathlib import Path" {
				continue
			}
			lines = append(lines, l)
		}
		parsed, err := godotenv.Unmarshal(strings.Join(lines, "\n"))
		if err != nil {
			return Config{}, fmt.Errorf("parse %s: %w", name, err)
		}
		for k, v := range parsed {
			values[k] = v
		}
	}
	get := func(k, d string) string {
		if v, ok := os.LookupEnv(k); ok {
			return v
		}
		if v, ok := values[k]; ok {
			return v
		}
		return d
	}
	var invalid []string
	number := func(k string, d, min, max int) int {
		v, e := strconv.Atoi(get(k, strconv.Itoa(d)))
		if e != nil || v < min || v > max {
			invalid = append(invalid, k)
			return d
		}
		return v
	}
	duration := func(k, d string) time.Duration {
		v, e := time.ParseDuration(get(k, d))
		if e != nil || v <= 0 {
			invalid = append(invalid, k)
		}
		return v
	}
	path := func(k, d string) string {
		p := get(k, d)
		// The copied .env contains absolute paths into the Python project.
		if i := strings.Index(filepath.ToSlash(p), "/rag-bench/"); i >= 0 {
			p = p[i+len("/rag-bench/"):]
		}
		if !filepath.IsAbs(p) {
			p = filepath.Join(root, p)
		}
		return filepath.Clean(p)
	}
	local := get("OPENAI_LOCAL_ENDPOINT", "http://127.0.0.1:1234/v1")
	remote := get("OPENAI_REMOTE_ENDPOINT", "http://192.168.1.50:1234/v1")
	key := get("OPENAI_API_KEY", "lm-studio")
	c := Config{Root: root, Listen: get("SERVER_ADDR", "127.0.0.1:8080"), APIKey: get("API_KEY", ""),
		S3Endpoint: get("S3_ENDPOINT_URL", ""), S3Region: get("S3_REGION", "us-east-1"), S3Bucket: get("S3_BUCKET", ""), S3Prefix: strings.Trim(get("S3_PATH_MARKDOWN_PREFIX", "markdown"), "/"), S3AccessKey: get("S3_ACCESS_KEY", ""), S3SecretKey: get("S3_SECRET_KEY", ""),
		QdrantHost: get("QDRANT_HOST", "192.168.1.50"), QdrantPort: number("QDRANT_GRPC_PORT", 6334, 1, 65535), QdrantKey: get("QDRANT_API_KEY", ""),
		Collection:     get("QDRANT_GO_COLLECTION", get("QDRANT_COLLECTION", "openrag")+"_go"),
		RerankEndpoint: get("RERANK_ENDPOINT", "http://192.168.1.50:8000/v1/rerank"), RerankModel: get("RERANK_MODEL", "jina-reranker-v3.5"), RerankKey: get("RERANK_API_KEY", ""),
		ChunkTokens: number("CHUNK_TOKENS", 500, 16, 8192), ChunkOverlap: number("CHUNK_OVERLAP", 50, 0, 8191), BatchSize: number("EMBEDDING_BATCH_SIZE", 16, 1, 256), RetrievalK: number("RETRIEVAL_K", 12, 1, 200), TopN: number("RERANK_TOP_N", 5, 1, 100), QueryVariations: number("QUERY_VARIATIONS", 3, 0, 10),
		HTTPTimeout: duration("HTTP_TIMEOUT", "120s"), RequestTimeout: duration("REQUEST_TIMEOUT", "15m"), MaxDocumentBytes: int64(number("MAX_DOCUMENT_BYTES", 32<<20, 1024, 256<<20)),
		QueriesFile: path("QUERIES_FILE", "openrag/queries.json"), AnswersFile: path("ANSWERS_FILE", "openrag/answers.json"), DatasetFile: path("DATASET_FILE", "results/results.jsonl"), OutputFile: path("OUTPUT_FILE", "results/output.jsonl"), ProgressFile: path("PROGRESS_FILE", "progress.json"), HallucinationsFile: path("HALLUCINATIONS_OUTPUT_FILE", "results/hallucinations.jsonl"), MissesFile: path("RETRIEVAL_MISSES_OUTPUT_FILE", "results/retrieval_misses.jsonl"), DashboardFile: path("DASHBOARD_FILE", "results/rag_metrics_dashboard.svg"),
	}
	c.QdrantTLS, err = strconv.ParseBool(get("QDRANT_TLS", "false"))
	if err != nil {
		invalid = append(invalid, "QDRANT_TLS")
	}
	c.Embedding = Model{get("EMBEDDING_ENDPOINT", remote), get("EMBEDDING_MODEL", "text-embedding-embeddinggemma-300m"), get("EMBEDDING_API_KEY", key)}
	c.RAG = Model{get("RAG_ENDPOINT", local), get("RAG_MODEL", "qwen2.5-7b-instruct-mlx"), get("RAG_API_KEY", key)}
	c.Judge = Model{get("EVAL_ENDPOINT", local), get("EVAL_MODEL", "meta-llama-3.1-8b-instruct"), get("EVAL_API_KEY", key)}
	c.Fusion = Model{get("FUSION_ENDPOINT", c.RAG.Endpoint), get("FUSION_MODEL", c.RAG.Name), get("FUSION_API_KEY", c.RAG.Key)}
	if c.ChunkOverlap >= c.ChunkTokens {
		invalid = append(invalid, "CHUNK_OVERLAP must be less than CHUNK_TOKENS")
	}
	if c.TopN > c.RetrievalK {
		invalid = append(invalid, "RERANK_TOP_N must be <= RETRIEVAL_K")
	}
	for k, v := range map[string]string{"EMBEDDING_ENDPOINT": c.Embedding.Endpoint, "RAG_ENDPOINT": c.RAG.Endpoint, "EVAL_ENDPOINT": c.Judge.Endpoint, "FUSION_ENDPOINT": c.Fusion.Endpoint, "RERANK_ENDPOINT": c.RerankEndpoint, "S3_ENDPOINT_URL": c.S3Endpoint} {
		if v == "" && k == "S3_ENDPOINT_URL" {
			continue
		}
		u, e := url.Parse(v)
		if e != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") {
			invalid = append(invalid, k)
		}
	}
	if c.Collection == "" {
		invalid = append(invalid, "QDRANT_GO_COLLECTION")
	}
	if len(invalid) > 0 {
		return Config{}, fmt.Errorf("invalid configuration: %s", strings.Join(invalid, ", "))
	}
	return c, nil
}
