package service

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"rag-bench-go/internal/chunker"
	"rag-bench-go/internal/config"
	"rag-bench-go/internal/domain"
	"rag-bench-go/internal/models"
	"rag-bench-go/internal/repository"
	"rag-bench-go/internal/sparse"
)

type chatFunc func(context.Context, string, string, float32) (string, error)

func (f chatFunc) Complete(c context.Context, s, u string, t float32) (string, error) {
	return f(c, s, u, t)
}

type embedFunc func(context.Context, []string) ([][]float32, error)

func (f embedFunc) Embed(c context.Context, s []string) ([][]float32, error) { return f(c, s) }

type rankFunc func(context.Context, string, []string, int) ([]models.Rank, error)

func (f rankFunc) Rerank(c context.Context, s string, d []string, n int) ([]models.Rank, error) {
	return f(c, s, d, n)
}

type memoryStore struct {
	docs     []domain.Chunk
	searches int
	upserts  int
	prunes   int
	ensured  bool
	fail     bool
}

func (s *memoryStore) Ensure(context.Context, int) error { s.ensured = true; return nil }
func (s *memoryStore) Upsert(_ context.Context, d []domain.Chunk, _ [][]float32) error {
	s.upserts++
	if s.fail {
		return errors.New("write failed")
	}
	s.docs = append(s.docs, d...)
	return nil
}
func (s *memoryStore) Prune(context.Context, string, []string) error { s.prunes++; return nil }
func (s *memoryStore) Search(context.Context, []float32, sparse.Vector, int) ([]domain.Chunk, error) {
	s.searches++
	return s.docs, nil
}
func embeddings(_ context.Context, s []string) ([][]float32, error) {
	out := make([][]float32, len(s))
	for i := range s {
		out[i] = []float32{1, 2}
	}
	return out, nil
}

func TestRAGOriginalQueryDedupTablesAndFallback(t *testing.T) {
	store := &memoryStore{docs: []domain.Chunk{{ID: "a", Text: "Prose", Metadata: map[string]any{"raw_table_content": []string{"| Model | Value |\n| --- | --- |\n| Gemma | 42 |"}}}, {ID: "b", Text: "Other evidence", Metadata: map[string]any{}}}}
	var captured string
	r := RAG{Store: store, Embedding: embedFunc(embeddings), Variations: 3, K: 12, TopN: 1, Fusion: chatFunc(func(context.Context, string, string, float32) (string, error) {
		return "1. alternate\n2. alternate\n3. Original", nil
	}), LLM: chatFunc(func(_ context.Context, system, prompt string, _ float32) (string, error) {
		captured = prompt
		return "42", nil
	}), Reranker: rankFunc(func(_ context.Context, _ string, docs []string, _ int) ([]models.Rank, error) {
		if len(docs) != 2 {
			t.Fatalf("dedup failed: %d", len(docs))
		}
		if !strings.Contains(docs[0], "Gemma") {
			t.Fatal("table missing from reranker context")
		}
		return nil, errors.New("offline")
	})}
	out, err := r.Answer(context.Background(), "Original")
	if err != nil {
		t.Fatal(err)
	}
	if store.searches != 2 || out.Queries[0] != "Original" {
		t.Fatalf("queries: %v", out.Queries)
	}
	if out.Answer != "42" || len(out.Warnings) != 1 || !strings.Contains(captured, "Gemma") {
		t.Fatalf("result: %+v", out)
	}
	store.docs = nil
	out, err = r.Answer(context.Background(), "Original")
	if err != nil || out.Answer != "I cannot find the answer in the provided documents." {
		t.Fatal("empty context did not abstain")
	}
}
func TestJudgeMetricsAndInvalidVerdicts(t *testing.T) {
	row := domain.Record{Question: "What?", Answer: "A. B.", GroundTruth: "A. C.", Contexts: []string{"A. Irrelevant."}}
	tests := []struct {
		name    string
		replies []string
		want    float64
	}{{"faithfulness", []string{"- A\n- B", "YES", "NO"}, .5}, {"context_recall", []string{"- A\n- C", "YES", "NO"}, .5}, {"answer_relevancy", []string{"What?", "0.8"}, .8}, {"context_precision", []string{"YES", "NO"}, .5}, {"answer_correctness", []string{"```json\n{\"TP\":[\"A\"],\"FP\":[\"B\"],\"FN\":[\"C\"]}\n```"}, .5}, {"answer_correctness", []string{"bad JSON", "0.25"}, .25}}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			i := 0
			e := Evaluator{Judge: chatFunc(func(context.Context, string, string, float32) (string, error) {
				if i >= len(tc.replies) {
					t.Fatal("unexpected judge call")
				}
				s := tc.replies[i]
				i++
				return s, nil
			})}
			v, err := e.Metric(context.Background(), tc.name, row)
			if err != nil || math.Abs(v-tc.want) > 1e-9 {
				t.Fatalf("got %v, %v", v, err)
			}
		})
	}
	e := Evaluator{Judge: chatFunc(func(context.Context, string, string, float32) (string, error) { return "NO, not YES", nil })}
	if _, err := e.yes(context.Background(), "check"); err == nil {
		t.Fatal("ambiguous verdict accepted")
	}
	for _, s := range []string{"NaN", "1.1", "-0.1", "score 0.5"} {
		e.Judge = chatFunc(func(context.Context, string, string, float32) (string, error) { return s, nil })
		if _, err := e.score(context.Background(), "score"); err == nil {
			t.Fatalf("accepted score %s", s)
		}
	}
	e.Judge = chatFunc(func(context.Context, string, string, float32) (string, error) {
		return "", errors.New("judge unavailable")
	})
	if _, err := e.Evaluate(context.Background(), row); err == nil {
		t.Fatal("upstream error silently converted to metrics")
	}
}

type memoryDocs map[string]string

func (d memoryDocs) List(context.Context, string) ([]string, error) {
	keys := []string{}
	for k := range d {
		keys = append(keys, k)
	}
	return keys, nil
}
func (d memoryDocs) Read(_ context.Context, k string) (string, error) { return d[k], nil }
func TestIndexPrunesOnlyAfterSuccessfulUpserts(t *testing.T) {
	c, _ := chunker.New(500, 50)
	store := &memoryStore{fail: true}
	index := Indexer{Documents: memoryDocs{"markdown/a.md": strings.Repeat("These scientific findings describe the system and explain several useful details for retrieval. ", 100)}, Store: store, Embedding: embedFunc(embeddings), Chunker: c, BatchSize: 1, Prefix: "markdown"}
	if _, err := index.Run(context.Background(), nil, 0, nil); err == nil {
		t.Fatal("write failure ignored")
	}
	if store.prunes != 0 {
		t.Fatal("old source pruned after failed write")
	}
	store.fail = false
	out, err := index.Run(context.Background(), nil, 0, nil)
	if err != nil || out.Chunks < 2 || store.prunes != 1 || !store.ensured {
		t.Fatalf("%+v %v", out, err)
	}
	if index.Allowed("other/a.md") || index.Allowed("markdown/a.pdf") {
		t.Fatal("out-of-scope key accepted")
	}
}
func TestEvaluationResumeAndAnalysis(t *testing.T) {
	root := t.TempDir()
	c := config.Config{DatasetFile: filepath.Join(root, "results.jsonl"), OutputFile: filepath.Join(root, "output.jsonl"), HallucinationsFile: filepath.Join(root, "hallucinations.jsonl"), MissesFile: filepath.Join(root, "misses.jsonl"), DashboardFile: filepath.Join(root, "dashboard.svg")}
	row := domain.Record{QueryID: "id", Question: "q", Answer: "a", GroundTruth: "a", Contexts: []string{"context"}}
	if err := repository.AppendJSONL(c.DatasetFile, row); err != nil {
		t.Fatal(err)
	}
	replies := []string{"- a", "YES", "q", "1", "- a", "YES", "YES", `{"TP":["a"],"FP":[],"FN":[]}`}
	i := 0
	e := &Evaluator{Judge: chatFunc(func(context.Context, string, string, float32) (string, error) {
		if i >= len(replies) {
			t.Fatal("unexpected judge call after resume")
		}
		s := replies[i]
		i++
		return s, nil
	})}
	b := Benchmark{Config: c, Evaluator: e}
	out, err := b.Evaluate(context.Background(), 0, nil)
	if err != nil || out.Processed != 1 {
		t.Fatalf("%+v %v", out, err)
	}
	out, err = b.Evaluate(context.Background(), 0, nil)
	if err != nil || out.Processed != 0 || out.Skipped != 1 {
		t.Fatalf("resume: %+v %v", out, err)
	}
	a, err := AnalyzeFile(c)
	if err != nil || a.Rows != 1 || a.Metrics["faithfulness"].Mean != 1 {
		t.Fatalf("%+v %v", a, err)
	}
	if err = ExportAnalysis(c, a); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{c.HallucinationsFile, c.MissesFile} {
		data, err := os.ReadFile(p)
		if err != nil || len(data) != 0 {
			t.Fatalf("empty failure export: %v", err)
		}
	}
}
func TestAnalysisStatistics(t *testing.T) {
	rows := []domain.Evaluated{}
	for _, v := range []float64{0, .5, 1} {
		rows = append(rows, domain.Evaluated{Metrics: domain.Metrics{Faithfulness: v, ContextRecall: v}})
	}
	a, err := Analyze(rows, .3, .3)
	if err != nil {
		t.Fatal(err)
	}
	s := a.Metrics["faithfulness"]
	if s.Mean != .5 || s.Std != .5 || s.Q1 != .25 || s.Q3 != .75 || len(a.Hallucinations) != 1 || len(a.RetrievalMisses) != 1 {
		t.Fatalf("%+v", a)
	}
}
func TestJobsConflictCancellationAndSnapshots(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	j := NewJobs(ctx)
	started := make(chan struct{})
	job, err := j.Start("test", func(ctx context.Context, p Progress) (any, error) {
		close(started)
		p(1, 2, "working")
		<-ctx.Done()
		return nil, ctx.Err()
	})
	if err != nil {
		t.Fatal(err)
	}
	<-started
	if _, err = j.Start("second", func(context.Context, Progress) (any, error) { return nil, nil }); !errors.Is(err, ErrBusy) {
		t.Fatal("concurrent job accepted")
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for range 100 {
				j.Get(job.ID)
			}
		})
	}
	if !j.Cancel(job.ID) {
		t.Fatal("cancel failed")
	}
	wg.Wait()
	j.Wait()
	final, ok := j.Get(job.ID)
	if !ok || final.Status != "cancelled" || final.Finished == nil {
		t.Fatalf("%+v", final)
	}
}

func TestBenchmarkResumesByIDAndRejectsChangedInput(t *testing.T) {
	root := t.TempDir()
	c := config.Config{QueriesFile: filepath.Join(root, "queries.json"), AnswersFile: filepath.Join(root, "answers.json"), DatasetFile: filepath.Join(root, "results.jsonl"), ProgressFile: filepath.Join(root, "progress.json")}
	if err := os.WriteFile(c.QueriesFile, []byte(`{"b":{"query":"second"},"a":{"query":"first"}}`), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(c.AnswersFile, []byte(`{"a":"first answer","b":"second answer"}`), 0600); err != nil {
		t.Fatal(err)
	}
	r := &RAG{Embedding: embedFunc(embeddings), Store: &memoryStore{}, K: 12, TopN: 5}
	b := Benchmark{Config: c, RAG: r}
	first, err := b.Run(context.Background(), 1, nil)
	if err != nil || first.Processed != 1 {
		t.Fatalf("%+v %v", first, err)
	}
	// A stale checkpoint must not re-run the already durable result.
	if err := os.WriteFile(c.ProgressFile, []byte(`{"last":-1}`), 0600); err != nil {
		t.Fatal(err)
	}
	second, err := b.Run(context.Background(), 0, nil)
	if err != nil || second.Processed != 1 || second.Skipped != 1 {
		t.Fatalf("%+v %v", second, err)
	}
	ids := []string{}
	if err := repository.ReadJSONL(c.DatasetFile, func(r domain.Record) error { ids = append(ids, r.QueryID); return nil }); err != nil {
		t.Fatal(err)
	}
	if strings.Join(ids, ",") != "a,b" {
		t.Fatalf("bad pairing/order: %v", ids)
	}
	if err := os.WriteFile(c.AnswersFile, []byte(`{"a":"changed","b":"second answer"}`), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := b.Run(context.Background(), 0, nil); err == nil {
		t.Fatal("changed input silently mixed into prior run")
	}
}
