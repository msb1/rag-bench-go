# Architecture

```text
cmd/rag-bench             CLI, server startup and graceful shutdown
internal/config          Legacy .env parsing, precedence, validation and path rebasing
internal/api             net/http routes, JSON validation, optional bearer authentication
internal/app             Constructs clients and injects service dependencies
internal/domain          Shared chunk/result/metric types and table cleanup
internal/chunker         Native token-bounded Markdown chunking
internal/sparse          Versioned native BM25 vocabulary and weights
internal/models          OpenAI-compatible SDK adapter and Jina HTTP client
internal/repository      RustFS S3, native Qdrant gRPC and durable JSONL files
internal/service         Indexing, RAG, judging, benchmark loops, analysis and jobs
resources                Embedded OpenAPI/Swagger plus example requests
```

Services depend on small interfaces (`Chat`, `Embedder`, `Reranker`, `Documents`, `VectorStore`). Constructors use a shared HTTP transport and an independent Qdrant connection pool. All upstream calls carry cancellation/deadline contexts. No orchestrator or general RAG framework is involved.

## Ingestion

1. List `.md`/`.markdown` keys under the configured RustFS prefix with an S3 paginator, or use explicit keys. Sort/deduplicate keys and apply an optional limit.
2. Read each UTF-8 object with a byte limit. Strip references outside code fences. Split on heading hierarchy and extract raw Markdown tables next to prose.
3. Split prose using the native `cl100k_base` tokenizer, preferring structural boundaries, with 500 tokens and 50-token overlap by default. Token boundaries are repaired to preserve UTF-8. Noise filtering follows the English prose rules of the reference implementation. Headers and table-only chunks survive appropriately.
4. Generate stable UUIDv5 IDs from `source__chunk__position__count`. Metadata includes source, hierarchy, logical ID, tokenizer, token count, type and raw tables.
5. Embed batches through the configured OpenAI-compatible Gemma endpoint using raw text strings. Validate returned indices, dimensions, finiteness and nonzero vectors. Never pre-tokenize strings into token IDs for the provider.
6. Infer dense dimension and create/validate Qdrant `dense` cosine and `sparse` IDF vectors. Use a dedicated collection. Check the existing schema marker before writes.
7. Upsert each batch with `wait=true`. Only after every batch succeeds, delete points for this source whose IDs are not in the current chunk set. There is no collection deletion or source-bucket mutation.

Qdrant point payloads retain the LangChain-like `page_content` / `metadata` shape plus a top-level `schema` marker. The marker records sparse vocabulary version, embedding model and chunking parameters. Source and schema fields get keyword indices. Sparse indices and values are supplied explicitly using the native driver; Qdrant computes IDF on query.

## Answer generation

```text
Question → RAG LLM query variations + original question
         → one embedding batch
         → per-query Qdrant dense/sparse RRF retrieval (k=12)
         → cross-query RRF, context deduplication
         → Jina reranking (top 5; table text included)
         → numbered context and grounding prompt
         → RAG LLM answer/summary plus source metadata
```

Cross-query RRF uses `1/(60 + one-based rank)`. Ties are sorted by document ID. Scores returned with final sources are reranker scores, or cross-query RRF scores if reranking failed. They are ranking signals, not calibrated probabilities. A reranker failure yields a response warning and the best fused results; cancelled calls propagate cancellation. Other provider/search failures propagate as errors. Empty retrieval bypasses answer generation and returns an abstention.

Tables are cleaned only when preparing context. Repeated table headers/continuation banners and HTML linebreak artifacts are normalized; the original Markdown remains in metadata. Empty cells and duplicated data rows are preserved.

## Evaluation, persistence and concurrency

`Evaluator.Metric` exposes all five metrics individually; `Evaluator.Evaluate` computes them sequentially. Every judge response is validated. Correctness classification accepts JSON inside Markdown fences; malformed classification JSON triggers the reference numeric fallback, while transport failures do not.

Benchmark input consists of query-ID-keyed query objects and answer strings. Completion records include query IDs and are fsynced before checkpoint updates. Restarting consults those records, tolerating an interruption after a durable result but before checkpoint update. Evaluation completion is keyed by SHA-256 of the canonical input record, so adding metrics does not change the identity. CSV/DataFrame processing has been replaced with streaming JSONL reads and native statistics.

The HTTP job registry runs one index/benchmark/eval job at a time, bounds retained jobs, and exposes progress/cancellation. Jobs use the server lifetime context rather than the initiating HTTP request's context. Per-service locks serialize file-writing workflows and source reindex/delete operations. Interactive RAG remains available during background work and has per-request state; no shared mutable “last contexts” object exists.

Ingestion replacement uses multiple acknowledged operations, not a transaction. A failure may leave partial new chunks until rerun. Job state itself is in memory. For multiple replicas or independent concurrent CLI writers, add distributed job coordination and a durable run database before sharing output files/collections.

## Upstream references

- [Qdrant native Go driver](https://github.com/qdrant/go-client)
- [Qdrant hybrid query documentation](https://qdrant.tech/documentation/search/hybrid-queries/)
- [go-openai SDK](https://github.com/sashabaranov/go-openai)
- [AWS SDK for Go v2](https://github.com/aws/aws-sdk-go-v2)
- [Native Go tokenizer](https://github.com/tiktoken-go/tokenizer)

The Python source remains the behavioral reference for metrics and Markdown/table handling. Intentional differences are documented in the main README: token sizes, complete heading support, table-only retention, original-query inclusion, real cross-query RRF, correct reranker index mapping, visible failures, native sparse vocabulary and resumable per-record output.
