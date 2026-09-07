package repository

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"unicode/utf8"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"rag-bench-go/internal/config"
)

type Documents interface {
	List(context.Context, string) ([]string, error)
	Read(context.Context, string) (string, error)
}
type S3 struct {
	client   *s3.Client
	bucket   string
	maxBytes int64
}

func NewS3(ctx context.Context, c config.Config, h *http.Client) (*S3, error) {
	cfg, err := awsconfig.LoadDefaultConfig(ctx, awsconfig.WithRegion(c.S3Region), awsconfig.WithHTTPClient(h), awsconfig.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(c.S3AccessKey, c.S3SecretKey, "")))
	if err != nil {
		return nil, err
	}
	client := s3.NewFromConfig(cfg, func(o *s3.Options) {
		o.UsePathStyle = true
		if c.S3Endpoint != "" {
			o.BaseEndpoint = aws.String(c.S3Endpoint)
		}
	})
	return &S3{client, c.S3Bucket, c.MaxDocumentBytes}, nil
}
func (s *S3) List(ctx context.Context, prefix string) ([]string, error) {
	if s.bucket == "" {
		return nil, fmt.Errorf("S3_BUCKET is required for ingestion")
	}
	if prefix != "" {
		prefix = strings.TrimRight(prefix, "/") + "/"
	}
	p := s3.NewListObjectsV2Paginator(s.client, &s3.ListObjectsV2Input{Bucket: aws.String(s.bucket), Prefix: aws.String(prefix)})
	out := []string{}
	for p.HasMorePages() {
		page, err := p.NextPage(ctx)
		if err != nil {
			return nil, fmt.Errorf("list S3 Markdown: %w", err)
		}
		for _, v := range page.Contents {
			key := aws.ToString(v.Key)
			lower := strings.ToLower(key)
			if strings.HasSuffix(lower, ".md") || strings.HasSuffix(lower, ".markdown") {
				out = append(out, key)
			}
		}
	}
	return out, nil
}
func (s *S3) Read(ctx context.Context, key string) (string, error) {
	r, err := s.client.GetObject(ctx, &s3.GetObjectInput{Bucket: aws.String(s.bucket), Key: aws.String(key)})
	if err != nil {
		return "", fmt.Errorf("read S3 object: %w", err)
	}
	defer r.Body.Close()
	if r.ContentLength != nil && *r.ContentLength > s.maxBytes {
		return "", fmt.Errorf("document exceeds MAX_DOCUMENT_BYTES")
	}
	b, err := io.ReadAll(io.LimitReader(r.Body, s.maxBytes+1))
	if err != nil {
		return "", err
	}
	if int64(len(b)) > s.maxBytes {
		return "", fmt.Errorf("document exceeds MAX_DOCUMENT_BYTES")
	}
	if !utf8.Valid(b) {
		return "", fmt.Errorf("document is not UTF-8")
	}
	return string(b), nil
}
