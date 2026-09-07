package models

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"rag-bench-go/internal/config"
)

type roundTrip func(*http.Request) (*http.Response, error)

func (f roundTrip) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
func reply(body string) *http.Response {
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"application/json"}}, Body: io.NopCloser(strings.NewReader(body))}
}
func TestEmbeddingsUseRawTextAndResponseIndices(t *testing.T) {
	h := &http.Client{Transport: roundTrip(func(r *http.Request) (*http.Response, error) {
		if r.URL.Path != "/v1/embeddings" || r.Header.Get("Authorization") != "Bearer secret" {
			t.Fatalf("bad request path/auth")
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body["model"] != "gemma" || body["input"].([]any)[0] != "raw markdown" {
			t.Fatalf("request body: %v", body)
		}
		return reply(`{"data":[{"index":1,"embedding":[3,4]},{"index":0,"embedding":[1,2]}]}`), nil
	})}
	m := NewOpenAI(config.Model{Endpoint: "http://model/v1", Name: "gemma", Key: "secret"}, h)
	v, err := m.Embed(context.Background(), []string{"raw markdown", "other"})
	if err != nil || v[0][0] != 1 || v[1][0] != 3 {
		t.Fatalf("%v %v", v, err)
	}
}
func TestRerankerSortsOriginalIndicesAndRejectsMalformed(t *testing.T) {
	for _, tc := range []struct {
		body string
		bad  bool
	}{{`{"results":[{"index":0,"relevance_score":0.1},{"index":1,"relevance_score":0.9}]}`, false}, {`{"results":[{"index":-1,"relevance_score":1}]}`, true}, {`{"results":[{"index":2,"relevance_score":1}]}`, true}, {`{"results":[{"index":0,"relevance_score":1},{"index":0,"relevance_score":0.5}]}`, true}, {`{"results":[{"relevance_score":1}]}`, true}, {`{"results":[]}`, true}} {
		j := Jina{Endpoint: "http://jina/v1/rerank", HTTP: &http.Client{Transport: roundTrip(func(r *http.Request) (*http.Response, error) { return reply(tc.body), nil })}}
		r, err := j.Rerank(context.Background(), "query", []string{"a", "b"}, 1)
		if (err != nil) != tc.bad {
			t.Fatalf("body %s: %v", tc.body, err)
		}
		if !tc.bad && (len(r) != 1 || r[0].Index != 1) {
			t.Fatalf("bad mapping: %+v", r)
		}
	}
}
func TestChatCancellationAndEmptyChoices(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	m := NewOpenAI(config.Model{Endpoint: "http://model/v1", Name: "test"}, &http.Client{Transport: roundTrip(func(r *http.Request) (*http.Response, error) { <-r.Context().Done(); return nil, r.Context().Err() })})
	if _, err := m.Complete(ctx, "", "query", 0); err == nil {
		t.Fatal("cancellation ignored")
	}
	m = NewOpenAI(config.Model{Endpoint: "http://model/v1", Name: "test"}, &http.Client{Transport: roundTrip(func(r *http.Request) (*http.Response, error) { return reply(`{"choices":[]}`), nil })})
	if _, err := m.Complete(context.Background(), "", "query", 0); err == nil {
		t.Fatal("empty choices accepted")
	}
}

func TestChatSendsExplicitTemperature(t *testing.T) {
	for _, temperature := range []float32{0, 0.7} {
		h := &http.Client{Transport: roundTrip(func(r *http.Request) (*http.Response, error) {
			var body map[string]json.RawMessage
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			if body["temperature"] == nil {
				t.Fatal("temperature omitted")
			}
			var got float32
			if err := json.Unmarshal(body["temperature"], &got); err != nil || got != temperature {
				t.Fatalf("temperature %s, wanted %v", body["temperature"], temperature)
			}
			return reply(`{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`), nil
		})}
		m := NewOpenAI(config.Model{Endpoint: "http://model/v1", Name: "test"}, h)
		if _, err := m.Complete(context.Background(), "", "query", temperature); err != nil {
			t.Fatal(err)
		}
	}
}
