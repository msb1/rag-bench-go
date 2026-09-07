package repository

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"rag-bench-go/internal/domain"
)

func TestJSONLCompatibilityAndTruncatedTail(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rows.jsonl")
	if err := os.WriteFile(path, []byte("{\"question\":\"q\",\"contexts\":\"one context\"}\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := ReadJSONL(path, func(r domain.Record) error {
		if len(r.Contexts) != 1 || r.Contexts[0] != "one context" {
			t.Fatal("legacy contexts not normalized")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if err := AppendJSONL(path, domain.Record{Question: "next", Contexts: []string{}}); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(`{"question":"incomplete`), 0600); err != nil {
		t.Fatal(err)
	}
	if err := AppendJSONL(path, domain.Record{}); err == nil {
		t.Fatal("appended to corrupt tail")
	}
	if err := ReadJSONL(path, func(domain.Record) error { return nil }); err == nil || !strings.Contains(err.Error(), "line 1") {
		t.Fatalf("lost parse location: %v", err)
	}
}
