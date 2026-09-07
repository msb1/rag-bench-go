package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"rag-bench-go/internal/api"
	"rag-bench-go/internal/app"
	"rag-bench-go/internal/config"
	"rag-bench-go/internal/domain"
	"rag-bench-go/internal/service"
)

func main() {
	if err := run(); err != nil {
		slog.Error("rag-bench", "error", err)
		os.Exit(1)
	}
}
func run() error {
	command := "server"
	args := os.Args[1:]
	if len(args) > 0 {
		command = args[0]
		args = args[1:]
	}
	fs := flag.NewFlagSet(command, flag.ContinueOnError)
	root := fs.String("root", ".", "project root containing .env")
	limit := fs.Int("limit", 0, "maximum new documents/rows (0 = all)")
	question := fs.String("question", "", "question for rag")
	key := fs.String("key", "", "one S3 Markdown object for chunker")
	file := fs.String("file", "", "JSON record for evaluate (default stdin)")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage: rag-bench <server|chunker|rag|benchmark|eval|evaluate|analyze|check> [flags]")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("unexpected arguments")
	}
	if *limit < 0 {
		return fmt.Errorf("limit must be nonnegative")
	}
	c, err := config.Load(*root)
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	a, err := app.New(ctx, c)
	if err != nil {
		return err
	}
	defer a.Close()
	progress := func(done, total int, detail string) {
		slog.Info("progress", "done", done, "total", total, "detail", detail)
	}
	print := func(v any) error { e := json.NewEncoder(os.Stdout); e.SetIndent("", "  "); return e.Encode(v) }
	switch command {
	case "server":
		srv := &http.Server{Addr: c.Listen, Handler: api.New(a), ReadHeaderTimeout: 10 * time.Second, ReadTimeout: 60 * time.Second, IdleTimeout: 120 * time.Second, WriteTimeout: c.RequestTimeout + 30*time.Second}
		done := make(chan error, 1)
		go func() {
			slog.Info("REST server starting", "address", c.Listen, "collection", c.Collection, "docs", "/docs")
			done <- srv.ListenAndServe()
		}()
		select {
		case err = <-done:
			if !errors.Is(err, http.ErrServerClosed) {
				return err
			}
		case <-ctx.Done():
			shutdown, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			if err = srv.Shutdown(shutdown); err != nil {
				_ = srv.Close()
				return err
			}
		}
		a.Jobs.Wait()
		return nil
	case "chunker":
		keys := []string{}
		if *key != "" {
			keys = append(keys, *key)
		}
		out, err := a.Indexer.Run(ctx, keys, *limit, progress)
		if err != nil {
			return err
		}
		return print(out)
	case "rag":
		if *question == "" {
			return fmt.Errorf("rag requires --question")
		}
		out, err := a.RAG.Answer(ctx, *question)
		if err != nil {
			return err
		}
		return print(out)
	case "benchmark":
		out, err := a.Benchmark.Run(ctx, *limit, progress)
		if err != nil {
			return err
		}
		return print(out)
	case "eval":
		out, err := a.Benchmark.Evaluate(ctx, *limit, progress)
		if err != nil {
			return err
		}
		return print(out)
	case "evaluate":
		var row domain.Record
		input := os.Stdin
		if *file != "" {
			input, err = os.Open(*file)
			if err != nil {
				return err
			}
			defer input.Close()
		}
		if err = json.NewDecoder(input).Decode(&row); err != nil {
			return err
		}
		out, err := a.Evaluator.Evaluate(ctx, row)
		if err != nil {
			return err
		}
		return print(out)
	case "analyze":
		out, err := service.AnalyzeFile(c)
		if err != nil {
			return err
		}
		if err = service.ExportAnalysis(c, out); err != nil {
			return err
		}
		return print(out)
	case "check":
		checkCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
		defer cancel()
		checks := map[string]any{}
		keys, s3err := a.S3.List(checkCtx, c.S3Prefix)
		if s3err != nil {
			checks["s3"] = s3err.Error()
		} else {
			checks["s3"] = map[string]int{"markdown_objects": len(keys)}
		}
		info, qerr := a.Store.Info(checkCtx)
		if qerr != nil {
			checks["qdrant"] = qerr.Error()
		} else {
			checks["qdrant"] = map[string]any{"collection": c.Collection, "points": info.GetPointsCount()}
		}
		if err = print(checks); err != nil {
			return err
		}
		if s3err != nil || qerr != nil {
			return fmt.Errorf("one or more checks failed (a new collection is created during ingestion)")
		}
		return nil
	default:
		fs.Usage()
		return fmt.Errorf("unknown command %q", command)
	}
}
