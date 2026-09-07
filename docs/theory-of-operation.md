# rag-bench-go theory of operation

## Applications and entry points

The project builds one executable, `bin/rag-bench`, whose first argument selects an application mode:

| Mode | Operation |
|---|---|
| `server` | Starts the HTTP API and waits for requests or shutdown. |
| `chunker` | Indexes selected or all RustFS Markdown objects into Qdrant. |
| `rag` | Retrieves evidence and generates one grounded answer. |
| `benchmark` | Generates OpenRAG answers and appends durable JSONL results. |
| `eval` | Evaluates benchmark JSONL rows through the judge model. |
| `evaluate` | Evaluates one JSON record from a file or stdin. |
| `analyze` | Computes statistics and exports failure files/dashboard. |
| `check` | Checks RustFS listing and Qdrant connectivity. |

The CLI constructs one `internal/app.App`: configuration, chunker, S3 repository, Qdrant repository, model clients, RAG service, evaluator, indexer, benchmark service, and in-memory job registry.

## Startup and configuration

`cmd/rag-bench/main.go` loads configuration from process environment, `.env.local`, `.env`, and defaults, in that order. It installs signal cancellation, builds the application, and closes idle HTTP/Qdrant resources on exit. Server mode binds loopback by default; `API_KEY` protects `/v1/` routes with a bearer token while health and documentation remain public.

## Indexing operation

The indexer lists and sorts Markdown keys, reads bounded UTF-8 S3 objects, performs heading-aware/table-aware token chunking, assigns stable UUID v5 IDs, embeds batches, and upserts dense/sparse Qdrant points. It prunes stale source points only after all current chunks are written. A missing collection is created using the embedding dimension; an existing schema is validated. Failed writes leave earlier data available for rerun.

## Retrieval and answer operation

The RAG service validates the question, generates up to three alternate queries, embeds the original plus variations, and performs Qdrant hybrid search for each. It fuses ranks with RRF, deduplicates contexts, sends the pool to Jina for top-five reranking, restores associated tables, and calls the answer model with a grounding-only prompt. Empty retrieval returns the explicit abstention. Reranker failure can fall back to fused order with a warning.

## Evaluation, benchmark, and analysis operation

The evaluator calls the judge model for five RAGAS-style text metrics and validates every response. The benchmark service reads query-ID-keyed OpenRAG data, appends each completed answer before updating its checkpoint, and resumes from durable rows. Evaluation resumes by a hash of the complete input record. Analysis computes metric summaries, classifies low-faithfulness and low-recall rows, and can write JSONL exports and an SVG dashboard.

## HTTP operation

The server exposes public `/healthz`, `/docs`, and `/openapi.json` plus versioned `/v1/` endpoints for configuration, chunk preview, indexing, RAG, retrieval, reranking, evaluation, benchmark/eval jobs, analysis, collection access, chunk lookup, and source deletion. Index, benchmark, and eval requests return `202` job snapshots. Only one background job runs at a time; poll `GET /v1/jobs/{id}` and cancel with `DELETE /v1/jobs/{id}`.

## Consistency and recovery

Jobs are in memory and disappear on restart; JSONL outputs and checkpoints are durable. The server uses request deadlines and propagates cancellation. Source replacement is acknowledged but not transactional, so readers can briefly observe mixed old/new points after a partial failure. Rerun indexing to converge, and use a fresh collection when embedding, tokenizer, chunking, or sparse vocabulary settings change.

## Run sequence

```sh
make check
make build
./bin/rag-bench check
./bin/rag-bench server
./bin/rag-bench chunker --limit 1
./bin/rag-bench benchmark --limit 3
./bin/rag-bench eval --limit 3
./bin/rag-bench analyze
```

The direct CLI commands and REST jobs use the same service implementations; only orchestration and progress reporting differ.
