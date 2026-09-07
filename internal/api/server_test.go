package api

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"rag-bench-go/internal/app"
	"rag-bench-go/internal/config"
	"rag-bench-go/resources"
)

func TestHTTPValidationAuthAndChunkPreview(t *testing.T) {
	c, err := config.Load(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	c.APIKey = "private-key"
	a, err := app.New(context.Background(), c)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	h := New(a)
	tests := []struct {
		method, path, body, auth string
		status                   int
	}{{"GET", "/healthz", "", "", 200}, {"GET", "/v1/config", "", "", 401}, {"GET", "/v1/config", "", "Bearer private-key", 200}, {"POST", "/v1/rag", `{"question":""}`, "Bearer private-key", 400}, {"POST", "/v1/rag", `{"question":"valid","typo":true}`, "Bearer private-key", 400}, {"POST", "/v1/rag", `{"question":"valid"} {}`, "Bearer private-key", 400}, {"POST", "/v1/evaluate/invalid", `{}`, "Bearer private-key", 404}, {"POST", "/v1/evaluate", `{"question":"q","answer":"a","ground_truth":"g"}`, "Bearer private-key", 400}, {"POST", "/v1/index", `{"keys":["elsewhere/file.md"]}`, "Bearer private-key", 400}, {"DELETE", "/v1/documents", "", "Bearer private-key", 400}, {"GET", "/v1/chunks/bad", "", "Bearer private-key", 400}, {"POST", "/v1/chunk", `{"key":"markdown/a.md","markdown":"# Heading\nThese are enough ordinary English words to preserve this useful scientific document in the vector index."}`, "Bearer private-key", 200}}
	for _, tc := range tests {
		req := httptest.NewRequest(tc.method, tc.path, strings.NewReader(tc.body))
		req.Header.Set("Authorization", tc.auth)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)
		if w.Code != tc.status {
			t.Fatalf("%s %s: got %d want %d: %s", tc.method, tc.path, w.Code, tc.status, w.Body.String())
		}
		if strings.Contains(w.Body.String(), "private-key") {
			t.Fatal("API key leaked")
		}
	}
}
func TestOpenAPISchemaReferences(t *testing.T) {
	var spec map[string]any
	if err := json.Unmarshal(resources.OpenAPI, &spec); err != nil {
		t.Fatal(err)
	}
	schemas := spec["components"].(map[string]any)["schemas"].(map[string]any)
	var walk func(any)
	walk = func(v any) {
		switch x := v.(type) {
		case map[string]any:
			if r, ok := x["$ref"].(string); ok {
				if _, found := schemas[strings.TrimPrefix(r, "#/components/schemas/")]; !found {
					t.Fatalf("unresolved schema %s", r)
				}
			}
			for _, v := range x {
				walk(v)
			}
		case []any:
			for _, v := range x {
				walk(v)
			}
		}
	}
	walk(spec)
	paths := spec["paths"].(map[string]any)
	for _, p := range []string{"/v1/chunk", "/v1/rag", "/v1/eval", "/v1/evaluate", "/v1/analyze", "/v1/jobs/{id}", "/v1/documents"} {
		if paths[p] == nil {
			t.Fatalf("missing docs for %s", p)
		}
	}
	if !strings.Contains(resources.Swagger, "/openapi.json") {
		t.Fatal("Swagger not wired to specification")
	}
}
