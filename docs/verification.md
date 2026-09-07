# Verification during the port

Environment: macOS arm64, Go 1.27.1. Checks performed September 6, 2026.

`make check` passed (race-enabled tests and `go vet`); `make build` and `go mod verify` passed. The compiled server was started on `127.0.0.1:8080`. Live HTTP checks returned 200 for `/healthz`, `/v1/config`, `/v1/chunk`, `/docs`, and `/openapi.json`. Chunk preview returned a deterministic UUID, heading/source metadata, and a measured 25-token prose chunk. The preview did not write to Qdrant.

## Live dependency checks

| Service | Observed result |
|---|---|
| RustFS configured bucket/prefix | Reachable; successfully listed **1,000 Markdown objects** with the copied credentials |
| Qdrant gRPC at the configured host/6334 | Reachable; returned `NotFound` for the new `openrag_go` collection, expected before ingestion |
| Embedding service, remote host port 1234 | Connection refused for `/v1/embeddings` and `/v1/models` |
| RAG/judge service, localhost port 1234 | Connection refused for `/v1/models` |
| Jina service, remote host port 8000 | Connection refused for `/openapi.json` |

A one-document ingestion was attempted for `markdown/2401.01872v2.md`. The source object was retrieved and chunking reached the embedding step. The refused embedding connection stopped processing **before collection creation or vector upsert**. No live vectors or benchmark scores were produced. The Python collection, S3 source data, and Python results were not modified.

To finish live validation, start the configured model servers or update `.env.local`, then run:

```sh
make build
./bin/rag-bench chunker --limit 1
./bin/rag-bench check
./bin/rag-bench rag --question "Summarize the main findings of the indexed paper."
./bin/rag-bench evaluate --file resources/example-evaluation.json
```

For a meaningful OpenRAG score, index the whole corpus before running benchmark/eval. A benchmark query whose source document is not indexed will legitimately abstain or have poor recall.

## Automated coverage

The project includes HTTP transport tests, an in-memory **actual gRPC protocol** test through the native Qdrant driver, and service/API regression tests. Controlled responses validate payloads, vector configurations, hybrid prefetch/RRF, source pruning, S3 signing/pagination, model response mapping, judge formulas and error behavior. These checks validate implementation contracts but do not substitute for a live model-quality benchmark.

Commands:

```sh
make check
make build
```
