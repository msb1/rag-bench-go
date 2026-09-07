package repository

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"rag-bench-go/internal/config"
)

type transportFunc func(*http.Request) (*http.Response, error)

func (f transportFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
func TestS3PaginationPathStyleAndLimits(t *testing.T) {
	calls := 0
	h := &http.Client{Transport: transportFunc(func(r *http.Request) (*http.Response, error) {
		calls++
		if !strings.HasPrefix(r.URL.Path, "/bucket") {
			t.Fatalf("not path style: %s", r.URL.Path)
		}
		if !strings.HasPrefix(r.Header.Get("Authorization"), "AWS4-HMAC-SHA256") {
			t.Fatal("request not SigV4 signed")
		}
		body := "markdown content"
		if r.URL.Query().Get("list-type") == "2" {
			if r.URL.Query().Get("prefix") != "markdown/" {
				t.Fatal("wrong prefix")
			}
			if r.URL.Query().Get("continuation-token") == "" {
				body = `<ListBucketResult><IsTruncated>true</IsTruncated><NextContinuationToken>next</NextContinuationToken><Contents><Key>markdown/a.md</Key></Contents><Contents><Key>markdown/a.pdf</Key></Contents></ListBucketResult>`
			} else {
				body = `<ListBucketResult><IsTruncated>false</IsTruncated><Contents><Key>markdown/b.MARKDOWN</Key></Contents></ListBucketResult>`
			}
		}
		return &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(body))}, nil
	})}
	s, err := NewS3(context.Background(), config.Config{S3Endpoint: "http://rustfs", S3Bucket: "bucket", S3Region: "us-east-1", S3AccessKey: "test", S3SecretKey: "test", MaxDocumentBytes: 10}, h)
	if err != nil {
		t.Fatal(err)
	}
	keys, err := s.List(context.Background(), "markdown")
	if err != nil || len(keys) != 2 || calls != 2 {
		t.Fatalf("pagination/filter: %v %v calls=%d", keys, err, calls)
	}
	if _, err = s.Read(context.Background(), "markdown/a.md"); err == nil {
		t.Fatal("oversized object accepted")
	}
}
