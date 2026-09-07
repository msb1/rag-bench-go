package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLegacyEnvironmentAndLocalOverrides(t *testing.T) {
	root := t.TempDir()
	data := `from pathlib import Path
S3_ENDPOINT_URL = "http://localhost:9000"
OPENAI_LOCAL_ENDPOINT = "http://localhost:1234/v1"
OPENAI_REMOTE_ENDPOINT = "http://remote:1234/v1"
RAG_MODEL = "rag"
EVAL_MODEL = "judge"
CODING_MODEL = "old-third-model"
QDRANT_COLLECTION = "openrag"
DATASET_FILE = "/Users/msb/Code/rag-bench/results/results.jsonl"
QUERIES_FILE = "/Users/msb/Code/rag-bench/openrag/queries.json"
`
	if err := os.WriteFile(filepath.Join(root, ".env"), []byte(data), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, ".env.local"), []byte("RERANK_TOP_N=4\n"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("RAG_MODEL", "process-rag")
	c, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if c.RAG.Name != "process-rag" || c.Fusion.Name != "process-rag" || c.Judge.Name != "judge" {
		t.Fatalf("model roles: %+v %+v", c.RAG, c.Fusion)
	}
	if c.Embedding.Endpoint != "http://remote:1234/v1" {
		t.Fatal("wrong embedding endpoint")
	}
	if c.Collection != "openrag_go" || c.TopN != 4 {
		t.Fatal("override failed")
	}
	if c.DatasetFile != filepath.Join(root, "results/results.jsonl") || c.QueriesFile != filepath.Join(root, "openrag/queries.json") {
		t.Fatal("legacy paths not rebased")
	}
}
func TestBadConfig(t *testing.T) {
	t.Setenv("CHUNK_TOKENS", "500")
	t.Setenv("CHUNK_OVERLAP", "500")
	if _, err := Load(t.TempDir()); err == nil {
		t.Fatal("accepted invalid overlap")
	}
}
