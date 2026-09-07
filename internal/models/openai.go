package models

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"regexp"
	"strings"

	openai "github.com/sashabaranov/go-openai"
	"rag-bench-go/internal/config"
)

type Chat interface {
	Complete(context.Context, string, string, float32) (string, error)
}
type Embedder interface {
	Embed(context.Context, []string) ([][]float32, error)
}
type OpenAI struct {
	client *openai.Client
	model  string
}

func NewOpenAI(c config.Model, h *http.Client) *OpenAI {
	cfg := openai.DefaultConfig(c.Key)
	cfg.BaseURL = strings.TrimRight(c.Endpoint, "/")
	cfg.HTTPClient = explicitTemperature{h}
	return &OpenAI{openai.NewClientWithConfig(cfg), c.Name}
}

// go-openai omits float32 zero via omitempty. Restore an explicit zero so judge
// and query-generation calls do not inherit the model server's temperature.
type explicitTemperature struct{ http *http.Client }

func (h explicitTemperature) Do(req *http.Request) (*http.Response, error) {
	if strings.HasSuffix(req.URL.Path, "/chat/completions") && req.Body != nil {
		body, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		_ = req.Body.Close()
		var fields map[string]json.RawMessage
		if err = json.Unmarshal(body, &fields); err != nil {
			return nil, err
		}
		if _, ok := fields["temperature"]; !ok {
			fields["temperature"] = json.RawMessage("0")
		}
		body, err = json.Marshal(fields)
		if err != nil {
			return nil, err
		}
		req = req.Clone(req.Context())
		req.Body = io.NopCloser(bytes.NewReader(body))
		req.ContentLength = int64(len(body))
		req.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(body)), nil }
	}
	return h.http.Do(req)
}

var thinking = regexp.MustCompile(`(?s)<think>.*?</think>`)

func CleanResponse(s string) string { return strings.TrimSpace(thinking.ReplaceAllString(s, "")) }
func (c *OpenAI) Complete(ctx context.Context, system, user string, temp float32) (string, error) {
	messages := []openai.ChatCompletionMessage{}
	if system != "" {
		messages = append(messages, openai.ChatCompletionMessage{Role: "system", Content: system})
	}
	messages = append(messages, openai.ChatCompletionMessage{Role: "user", Content: user})
	r, err := c.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{Model: c.model, Messages: messages, Temperature: temp})
	if err != nil {
		return "", fmt.Errorf("chat completion (%s): %w", c.model, err)
	}
	if len(r.Choices) == 0 {
		return "", fmt.Errorf("chat completion returned no choices")
	}
	s := CleanResponse(r.Choices[0].Message.Content)
	if s == "" {
		return "", fmt.Errorf("chat completion returned empty content")
	}
	return s, nil
}
func (c *OpenAI) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}
	r, err := c.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{Model: openai.EmbeddingModel(c.model), Input: texts, EncodingFormat: openai.EmbeddingEncodingFormatFloat})
	if err != nil {
		return nil, fmt.Errorf("embeddings: %w", err)
	}
	if len(r.Data) != len(texts) {
		return nil, fmt.Errorf("embedding count mismatch: got %d, want %d", len(r.Data), len(texts))
	}
	out := make([][]float32, len(texts))
	dim := 0
	for _, d := range r.Data {
		if d.Index < 0 || d.Index >= len(out) || out[d.Index] != nil {
			return nil, fmt.Errorf("invalid embedding response index")
		}
		if len(d.Embedding) == 0 {
			return nil, fmt.Errorf("empty embedding")
		}
		if dim == 0 {
			dim = len(d.Embedding)
		}
		if dim != len(d.Embedding) {
			return nil, fmt.Errorf("inconsistent embedding dimensions")
		}
		norm := 0.0
		for _, v := range d.Embedding {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("nonfinite embedding")
			}
			norm += float64(v) * float64(v)
		}
		if norm == 0 {
			return nil, fmt.Errorf("zero embedding")
		}
		out[d.Index] = d.Embedding
	}
	return out, nil
}
