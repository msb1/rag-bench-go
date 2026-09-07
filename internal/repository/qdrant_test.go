package repository

import (
	"context"
	"net"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	q "github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/test/bufconn"
	"rag-bench-go/internal/domain"
	"rag-bench-go/internal/sparse"
)

type collectionRPC struct {
	q.UnimplementedCollectionsServer
	mu      sync.Mutex
	created *q.CreateCollection
}

func (c *collectionRPC) CollectionExists(context.Context, *q.CollectionExistsRequest) (*q.CollectionExistsResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return &q.CollectionExistsResponse{Result: &q.CollectionExists{Exists: c.created != nil}}, nil
}
func (c *collectionRPC) Create(_ context.Context, r *q.CreateCollection) (*q.CollectionOperationResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.created = r
	return &q.CollectionOperationResponse{Result: true}, nil
}
func (c *collectionRPC) Get(context.Context, *q.GetCollectionInfoRequest) (*q.GetCollectionInfoResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return &q.GetCollectionInfoResponse{Result: &q.CollectionInfo{Config: &q.CollectionConfig{Params: &q.CollectionParams{VectorsConfig: c.created.VectorsConfig, SparseVectorsConfig: c.created.SparseVectorsConfig}}}}, nil
}

type pointsRPC struct {
	q.UnimplementedPointsServer
	mu      sync.Mutex
	upsert  *q.UpsertPoints
	query   *q.QueryPoints
	deleted *q.DeletePoints
}

func (p *pointsRPC) Upsert(_ context.Context, r *q.UpsertPoints) (*q.PointsOperationResponse, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.upsert = r
	return &q.PointsOperationResponse{Result: &q.UpdateResult{Status: q.UpdateStatus_Completed}}, nil
}
func (p *pointsRPC) Query(_ context.Context, r *q.QueryPoints) (*q.QueryResponse, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.query = r
	c := p.upsert.Points[0]
	return &q.QueryResponse{Result: []*q.ScoredPoint{{Id: c.Id, Payload: c.Payload, Score: 0.9}}}, nil
}
func (p *pointsRPC) Scroll(context.Context, *q.ScrollPoints) (*q.ScrollResponse, error) {
	return &q.ScrollResponse{}, nil
}
func (p *pointsRPC) CreateFieldIndex(context.Context, *q.CreateFieldIndexCollection) (*q.PointsOperationResponse, error) {
	return &q.PointsOperationResponse{Result: &q.UpdateResult{Status: q.UpdateStatus_Completed}}, nil
}
func (p *pointsRPC) Delete(_ context.Context, r *q.DeletePoints) (*q.PointsOperationResponse, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.deleted = r
	return &q.PointsOperationResponse{Result: &q.UpdateResult{Status: q.UpdateStatus_Completed}}, nil
}

func TestNativeQdrantHybridRoundTrip(t *testing.T) {
	listener := bufconn.Listen(1 << 20)
	server := grpc.NewServer()
	collections := &collectionRPC{}
	points := &pointsRPC{}
	q.RegisterCollectionsServer(server, collections)
	q.RegisterPointsServer(server, points)
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(server.Stop)
	client, err := q.NewClient(&q.Config{Host: "127.0.0.1", SkipCompatibilityCheck: true, PoolSize: 1, GrpcOptions: []grpc.DialOption{grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return listener.DialContext(ctx) })}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = client.Close() })
	r := Qdrant{client: client, collection: "test_go", schema: "test-schema", timeout: 5 * time.Second}
	ctx := context.Background()
	if err = r.Ensure(ctx, 3); err != nil {
		t.Fatal(err)
	}
	collections.mu.Lock()
	sp := collections.created.SparseVectorsConfig.GetMap()["sparse"]
	if sp.GetModifier() != q.Modifier_Idf {
		t.Fatal("missing IDF")
	}
	collections.mu.Unlock()
	table := "| Item | Result |\n| --- | --- |\n| Test | 42 |"
	chunk := domain.Chunk{ID: uuid.NewString(), Text: "Native hybrid retrieval", Metadata: map[string]any{"source": "markdown/a.md", "raw_table_content": []string{table}, "has_tables": true}}
	if err = r.Upsert(ctx, []domain.Chunk{chunk}, [][]float32{{1, 2, 3}}); err != nil {
		t.Fatal(err)
	}
	out, err := r.Search(ctx, []float32{1, 2, 3}, sparse.Query("result"), 12)
	if err != nil || len(out) != 1 || out[0].ID != chunk.ID || !strings.Contains(out[0].Context(), table) {
		t.Fatalf("payload roundtrip: %+v %v", out, err)
	}
	points.mu.Lock()
	if len(points.query.Prefetch) != 2 || points.query.Query.GetFusion() != q.Fusion_RRF || points.query.Prefetch[0].GetUsing() != "dense" || points.query.Prefetch[1].GetUsing() != "sparse" {
		t.Fatal("not a dense/sparse RRF query")
	}
	if len(points.query.Prefetch[0].Filter.Must) != 1 || !points.upsert.GetWait() {
		t.Fatal("schema filter or write acknowledgement missing")
	}
	points.mu.Unlock()
	if _, err = r.Search(ctx, []float32{1, 2, 3}, sparse.Query("the"), 12); err != nil {
		t.Fatal(err)
	}
	points.mu.Lock()
	if len(points.query.Prefetch) != 0 || points.query.GetUsing() != "dense" {
		t.Fatal("empty sparse query should use dense search")
	}
	points.mu.Unlock()
	if err = r.Prune(ctx, "markdown/a.md", []string{chunk.ID}); err != nil {
		t.Fatal(err)
	}
	points.mu.Lock()
	f := points.deleted.Points.GetFilter()
	if len(f.Must) != 1 || len(f.MustNot) != 1 || f.MustNot[0].GetHasId().HasId[0].GetUuid() != chunk.ID {
		t.Fatal("prune must exclude all current chunk IDs")
	}
	points.mu.Unlock()
	if err = r.Ensure(ctx, 99); err == nil {
		t.Fatal("incompatible dimensions accepted")
	}
}
