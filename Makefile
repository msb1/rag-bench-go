GOCACHE ?= $(CURDIR)/.cache/build
export GOCACHE

.PHONY: build run test vet check
build:
	go build -trimpath -o bin/rag-bench ./cmd/rag-bench
run:
	go run ./cmd/rag-bench server
test:
	go test -race ./...
vet:
	go vet ./...
check: test vet
