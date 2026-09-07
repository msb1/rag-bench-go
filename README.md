# rag-bench-go

Documentation map: [architecture](docs/architecture.md), [theory of operation](docs/theory-of-operation.md), and [verification](docs/verification.md).

Native Go rewrite of `rag-bench`: RustFS Markdown ingestion, 500-token chunking, hybrid Qdrant retrieval, Jina reranking, grounded LLM answers/summaries, and resumable Open RAG Benchmark evaluation. Uses `net/http`, provider SDKs, and small repository/service interfaces. No LangChain, Python process, or GPU runtime is needed in this application.

## Start the server

Requires Go 1.26 or newer and reachable RustFS, Qdrant **gRPC port 6334**, embedding, RAG, judge, and Jina services.

```sh
cd /Users/msb/Code/rag-bench-go
make build
./bin/rag-bench server
```

Open **http://127.0.0.1:8080/docs** for Swagger UI. The embedded OpenAPI 3.0.3 document is at `/openapi.json` and [resources/openapi.json](resources/openapi.json). Swagger UI loads pinned JavaScript/CSS from jsDelivr; the JSON specification and all APIs work without that CDN.

The original `.env` has already been copied, byte for byte, with mode `0600`, and is ignored by Git. The loader accepts its spaces around `=` and ignores its stray `from pathlib import Path` line. It rebases legacy absolute `/rag-bench/` dataset/output paths to this project's root. The OpenRAG dataset has also been copied with its original license notice. Python results and checkpoints were not copied.

Configuration precedence is **process environment > `.env.local` > `.env` > defaults**. For overrides, create `.env.local` or export environment variables. See [.env.example](.env.example) for all supported settings. Run from this directory, or pass `--root /path/to/rag-bench-go` after the command. Configuration is loaded at startup.

## Model and storage configuration

| Role | Configuration | Default used with the copied `.env` |
|---|---|---|
| Dense embeddings | `EMBEDDING_MODEL`, `EMBEDDING_ENDPOINT` | Gemma from `.env`; endpoint defaults to `OPENAI_REMOTE_ENDPOINT` |
| Answer/summary LLM | `RAG_MODEL`, `RAG_ENDPOINT` | RAG model from `.env`; `OPENAI_LOCAL_ENDPOINT` |
| Query variations | `FUSION_MODEL`, `FUSION_ENDPOINT` | Same model and endpoint as RAG |
| Judge LLM | `EVAL_MODEL`, `EVAL_ENDPOINT` | Evaluation model from `.env`; `OPENAI_LOCAL_ENDPOINT` |
| Jina reranker | `RERANK_MODEL`, `RERANK_ENDPOINT` | `jina-reranker-v3.5`; `http://192.168.1.50:8000/v1/rerank` |
| Qdrant | `QDRANT_HOST`, `QDRANT_GRPC_PORT` | `192.168.1.50:6334` |
| Go collection | `QDRANT_GO_COLLECTION` | `QDRANT_COLLECTION` + `_go`, currently `openrag_go` |
| Markdown documents | Existing `S3_*` variables | Configured RustFS bucket and `markdown/` prefix |

The old Python embedding, Qdrant and reranker addresses were hardcoded outside `.env`; the Go version exposes them as configuration. The legacy `CODING_MODEL` setting is accepted but not used automatically: query variations share the RAG LLM, so only the requested two chat models are needed. Set `FUSION_MODEL` and `FUSION_ENDPOINT` if a separate query model is desired.

OpenAI-compatible base URLs include `/v1`. `RERANK_ENDPOINT` is the **complete** rerank URL. Use `OPENAI_API_KEY` or per-role `EMBEDDING_API_KEY`, `RAG_API_KEY`, `EVAL_API_KEY`, and `FUSION_API_KEY`. Jina uses optional `RERANK_API_KEY`. RustFS uses SigV4 signing and path-style S3 addressing. Qdrant supports `QDRANT_API_KEY` and `QDRANT_TLS=true`.

The Go service calls the remote Jina inference service; it does not port the Python Transformers/GPU model host into Go. The rerank response must contain the **original input document indices**, not enumeration positions after sorting. This fixes an assumption violated by the original Python server's result mapping.

## Index, ask, and evaluate

Start the model services first, then make a small indexing pass:

```sh
./bin/rag-bench check
./bin/rag-bench chunker --limit 1
./bin/rag-bench rag --question "Summarize the main findings of the indexed paper."
```

`check` only inspects RustFS and the configured collection. A missing Go collection is expected before first ingestion. Ingestion infers dense dimensions from the embedding response, creates cosine `dense` and IDF-enabled `sparse` vectors if needed, and validates existing collection compatibility. Once the small pass succeeds, index the whole bucket with `./bin/rag-bench chunker`.

With the server running, the equivalent calls are:

```sh
curl -sS http://127.0.0.1:8080/v1/index \
  -H 'Content-Type: application/json' -d '{"limit":1}'

# Substitute the id from the 202 response; wait for status "completed".
curl -sS http://127.0.0.1:8080/v1/jobs/JOB_ID

curl -sS http://127.0.0.1:8080/v1/rag \
  -H 'Content-Type: application/json' \
  -d '{"question":"Summarize the main findings of the indexed paper."}'

curl -sS http://127.0.0.1:8080/v1/evaluate \
  -H 'Content-Type: application/json' \
  --data-binary @resources/example-evaluation.json
```

The RAG response contains `answer`, `contexts`, `sources` with IDs/metadata, generated `queries`, and any fallback `warnings`. Asking for a summary uses the same grounded answer endpoint. No retrieved context produces the explicit abstention: “I cannot find the answer in the provided documents.” The RAG endpoint does not append interactive requests to benchmark files.

Run the OpenRAG workflow after indexing the corpus:

```sh
./bin/rag-bench benchmark --limit 3
./bin/rag-bench eval --limit 3
./bin/rag-bench analyze
```

Omit `--limit` for all remaining records. `benchmark` generates answers from `QUERIES_FILE` and `ANSWERS_FILE`; `eval` judges `DATASET_FILE`. `evaluate --file resources/example-evaluation.json` evaluates one record and prints JSON. `evaluate` also accepts JSON on stdin.

## Python-to-Go workflow mapping

| Python application | Go CLI | REST API |
|---|---|---|
| `chunker.py` | `chunker [--key markdown/paper.md] [--limit N]` | `POST /v1/index`; preview with `POST /v1/chunk` |
| `rag.py`, interactive | `rag --question "..."` | `POST /v1/rag`; retrieval only at `/v1/retrieve` |
| `rag.py`, benchmark loop | `benchmark [--limit N]` | `POST /v1/benchmark` |
| `eval.py` | `eval [--limit N]` | `POST /v1/eval` |
| `evaluate.py` | `evaluate [--file record.json]` | `POST /v1/evaluate`, `/v1/evaluate/{metric}` |
| `analyze.py` | `analyze` | `POST /v1/analyze`; read-only `/v1/analysis`, `/v1/dashboard.svg` |
| Reranker client | Used by RAG | `POST /v1/rerank` |
| Qdrant helpers | Used by ingestion and retrieval | `GET/POST /v1/collection`, `GET /v1/chunks/{id}`, `DELETE /v1/documents?key=...` |

Conversion and S3 migration are intentionally excluded: documents must already be Markdown in RustFS.

## Chunking and retrieval behavior

The native chunker strips references/bibliography/works-cited sections, preserves Markdown headings and their hierarchy, extracts tables into `metadata.raw_table_content`, recursively splits prose at paragraph/line/word boundaries, and filters tiny/noisy/math-only fragments. It retains table-only sections and recognizes headings `#` through `######`; fenced code is not treated as section headers. UUIDv5 logical chunk IDs remain deterministic for the source key, chunk position and count.

The strict prose limit is **500 `cl100k_base` BPE tokens**, with up to **50 tokens of overlap within sections**, configurable via `CHUNK_TOKENS` and `CHUNK_OVERLAP`. This is a real native token count, not characters or whitespace words. It is a stable chunking tokenizer, **not Gemma's SentencePiece tokenizer**; model-specific token counts will differ. Large raw tables remain whole in metadata and are outside the prose token limit, matching the table-preservation strategy. Allow sufficient LLM context for five chunks plus associated tables and judge prompts. Non-English fragments may still be removed by the inherited English-word noise filter.

Retrieval searches the original question plus up to three query variations. Each query uses Qdrant dense/sparse prefetches with reciprocal-rank fusion (RRF), then the service fuses query result lists, deduplicates context, and calls Jina once for the top five passages. Table text participates in sparse matching and reranking and is included in the final LLM context. Dense embeddings use prose; table-only chunks use a small placeholder for the dense vector and their actual tables for sparse matching.

Native BM25 uses Unicode words, English stopword filtering, stable FNV-1a indices, `k1=1.2`, `b=0.75`, and reference average length 256. Query weights are 1; Qdrant supplies corpus IDF. This vocabulary is **not FastEmbed `Qdrant/bm25` compatible**. The separate `openrag_go` collection avoids mixing the two. Reindex Markdown rather than reusing Python vectors. Changing the embedding model, tokenizer settings or sparse version requires a fresh compatible collection. See [docs/architecture.md](docs/architecture.md).

## Evaluation and output

These reproduce the project's **RAGAS-style text-judge metrics**, not the official RAGAS library implementation:

| Metric | Computation |
|---|---|
| `faithfulness` | Answer claims supported by context / all extracted answer claims |
| `answer_relevancy` | Reverse-generate the question, then judge semantic intent similarity |
| `context_recall` | Ground-truth facts supported by context / all ground-truth facts |
| `context_precision` | Useful context sentences / all context sentences, preserving Python's sentence-level metric |
| `answer_correctness` | `TP / (TP + 0.5 * (FP + FN))`; numeric judgment fallback only for malformed classification JSON |

Judge API failures, ambiguous verdicts, missing scores and out-of-range values are errors. They are not recorded as zero-quality answers. Query-generation and judge temperatures are explicitly zero; answer generation uses 0.7. Empty claim lists and empty context precision denominators retain the Python score of zero.

| Default output | Contents |
|---|---|
| `results/results.jsonl` | Generated benchmark answers, contexts, source metadata and query IDs |
| `results/output.jsonl` | Original records plus all five metrics |
| `progress.json` | Last completed query ID; informational checkpoint |
| `results/hallucinations.jsonl` | Faithfulness strictly below 0.30 |
| `results/retrieval_misses.jsonl` | Context recall strictly below 0.30 |
| `results/rag_metrics_dashboard.svg` | Standalone mean bars and quartile/min/max distributions |

Resume uses durable JSONL rows, not a fragile numeric index. Each completed row is flushed to disk. Queries are processed in sorted ID order, so order differs from Python without changing dataset pairing. Evaluation resumes by the hash of the complete input record. A changed benchmark question/ground truth under an existing ID is rejected; use new output files for a new experiment or changed model configuration. Malformed/truncated JSONL stops with its line number; repair the final row before resuming. Failure exports are rewritten even when empty, preventing stale failures from earlier runs.

## Jobs and operation

`POST /v1/index`, `/v1/benchmark`, and `/v1/eval` accept JSON and return **202** with a job ID. Poll `GET /v1/jobs/{id}`; request cancellation with `DELETE /v1/jobs/{id}`. One background job runs at a time; another start returns **409**. The last 100 jobs are held in memory; restarting clears job history but keeps durable benchmark/evaluation results. Restart ingestion to reprocess source objects with stable IDs. There is no persistent ingestion skip cache.

Reindexing upserts new chunks first, then removes stale chunks for that exact source. Failed batches leave the previous data recoverable by a rerun. Replacement is not transactionally atomic: readers may temporarily see partial new and old chunks. Empty/noise-only documents are skipped without deleting existing chunks; use the explicit source-delete endpoint when deletion is intended. Source deletion only removes Qdrant points, not the S3 object. The server serializes source writes; run only one writer process per collection and output set.

The server defaults to loopback. Set `SERVER_ADDR` to change it and `API_KEY` to require `Authorization: Bearer <key>` for every `/v1/` endpoint. `/healthz`, `/docs`, and `/openapi.json` remain public. `/healthz` means the process is alive, not that models are ready. Upstream operations have `HTTP_TIMEOUT` (120s), synchronous requests have `REQUEST_TIMEOUT` (15m), and jobs support cancellation. Request bodies are capped at 32 MiB and S3 objects at `MAX_DOCUMENT_BYTES` (32 MiB). Logs include operations and timing, not `.env` contents or prompts.

## Development and verification

```sh
make check   # race-enabled tests and go vet
make build
```

Tests cover token limits, Unicode preservation, tables, deterministic IDs, configuration migration, S3 pagination/SigV4, raw embedding requests, explicit zero temperature, reranker indices, native Qdrant gRPC hybrid queries and pruning, RAG fusion, judge arithmetic/error handling, JSONL resume, job cancellation and REST validation. Unit/integration tests use controlled transports and in-memory gRPC; they do not require live models.

See [docs/verification.md](docs/verification.md) for the live-service checks performed during the port. The existing source project and Python collection are unchanged. Application code retains the source MIT license; the copied OpenRAG dataset retains its separate CC-BY-NC-4.0 notice in [openrag/README.md](openrag/README.md).
