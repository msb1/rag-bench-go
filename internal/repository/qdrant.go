package repository

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	q "github.com/qdrant/go-client/qdrant"
	"rag-bench-go/internal/config"
	"rag-bench-go/internal/domain"
	"rag-bench-go/internal/sparse"
)

type VectorStore interface {
	Ensure(context.Context, int) error
	Upsert(context.Context, []domain.Chunk, [][]float32) error
	Prune(context.Context, string, []string) error
	Search(context.Context, []float32, sparse.Vector, int) ([]domain.Chunk, error)
}
type Qdrant struct {
	client             *q.Client
	collection, schema string
	timeout            time.Duration
}

func NewQdrant(c config.Config) (*Qdrant, error) {
	client, err := q.NewClient(&q.Config{Host: c.QdrantHost, Port: c.QdrantPort, APIKey: c.QdrantKey, UseTLS: c.QdrantTLS, SkipCompatibilityCheck: true})
	if err != nil {
		return nil, err
	}
	return &Qdrant{client, c.Collection, fmt.Sprintf("%s|%s|cl100k_base|%d|%d", sparse.Version, c.Embedding.Name, c.ChunkTokens, c.ChunkOverlap), c.HTTPTimeout}, nil
}
func (r *Qdrant) Close() error { return r.client.Close() }
func (r *Qdrant) Ensure(ctx context.Context, dimension int) error {
	ctx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	exists, err := r.client.CollectionExists(ctx, r.collection)
	if err != nil {
		return fmt.Errorf("Qdrant collection lookup: %w", err)
	}
	if !exists {
		err = r.client.CreateCollection(ctx, &q.CreateCollection{CollectionName: r.collection, VectorsConfig: q.NewVectorsConfigMap(map[string]*q.VectorParams{"dense": {Size: uint64(dimension), Distance: q.Distance_Cosine}}), SparseVectorsConfig: q.NewSparseVectorsConfig(map[string]*q.SparseVectorParams{"sparse": {Modifier: q.PtrOf(q.Modifier_Idf)}})})
		if err != nil {
			return fmt.Errorf("create hybrid collection: %w", err)
		}
	}
	info, err := r.client.GetCollectionInfo(ctx, r.collection)
	if err != nil {
		return err
	}
	params := info.GetConfig().GetParams()
	dense := params.GetVectorsConfig().GetParamsMap().GetMap()["dense"]
	sp := params.GetSparseVectorsConfig().GetMap()["sparse"]
	if dense == nil || dense.GetSize() != uint64(dimension) || dense.GetDistance() != q.Distance_Cosine || sp == nil || sp.GetModifier() != q.Modifier_Idf {
		return fmt.Errorf("collection must have dense cosine vectors of size %d and sparse vectors with IDF; choose a new QDRANT_GO_COLLECTION", dimension)
	}
	points, err := r.client.Scroll(ctx, &q.ScrollPoints{CollectionName: r.collection, Limit: q.PtrOf(uint32(1)), WithPayload: q.NewWithPayload(true)})
	if err != nil {
		return err
	}
	if len(points) > 0 && points[0].Payload["schema"].GetStringValue() != r.schema {
		return fmt.Errorf("collection contains an incompatible index; choose a new QDRANT_GO_COLLECTION")
	}
	for _, field := range []string{"metadata.source", "schema"} {
		_, err = r.client.CreateFieldIndex(ctx, &q.CreateFieldIndexCollection{CollectionName: r.collection, FieldName: field, FieldType: q.PtrOf(q.FieldType_FieldTypeKeyword), Wait: q.PtrOf(true)})
		if err != nil {
			return err
		}
	}
	return nil
}
func (r *Qdrant) Upsert(ctx context.Context, chunks []domain.Chunk, embeddings [][]float32) error {
	if len(chunks) != len(embeddings) {
		return fmt.Errorf("chunk and embedding count mismatch")
	}
	if len(chunks) == 0 {
		return nil
	}
	ctx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	points := make([]*q.PointStruct, len(chunks))
	for i, c := range chunks {
		s := sparse.Document(c.Context())
		payload, err := chunkPayload(c, r.schema)
		if err != nil {
			return err
		}
		points[i] = &q.PointStruct{Id: q.NewIDUUID(c.ID), Vectors: q.NewVectorsMap(map[string]*q.Vector{"dense": q.NewVectorDense(embeddings[i]), "sparse": q.NewVectorSparse(s.Indices, s.Values)}), Payload: payload}
	}
	_, err := r.client.Upsert(ctx, &q.UpsertPoints{CollectionName: r.collection, Points: points, Wait: q.PtrOf(true)})
	return err
}

func chunkPayload(c domain.Chunk, schema string) (map[string]*q.Value, error) {
	// Normalize typed slices (such as []string tables) to protobuf-compatible JSON values.
	data, err := json.Marshal(map[string]any{"page_content": c.Text, "metadata": c.Metadata, "schema": schema})
	if err != nil {
		return nil, err
	}
	var payload map[string]any
	if err = json.Unmarshal(data, &payload); err != nil {
		return nil, err
	}
	return q.TryValueMap(payload)
}
func sourceFilter(source string) *q.Filter {
	return &q.Filter{Must: []*q.Condition{q.NewMatch("metadata.source", source)}}
}

// Prune runs only after every new batch is acknowledged, so an embedding failure
// cannot delete the previous source. Retried ingestion repairs partial upserts.
func (r *Qdrant) Prune(ctx context.Context, source string, keep []string) error {
	ctx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	filter := sourceFilter(source)
	if len(keep) > 0 {
		ids := make([]*q.PointId, len(keep))
		for i, id := range keep {
			ids[i] = q.NewIDUUID(id)
		}
		filter.MustNot = []*q.Condition{{ConditionOneOf: &q.Condition_HasId{HasId: &q.HasIdCondition{HasId: ids}}}}
	}
	_, err := r.client.Delete(ctx, &q.DeletePoints{CollectionName: r.collection, Points: q.NewPointsSelectorFilter(filter), Wait: q.PtrOf(true)})
	return err
}
func (r *Qdrant) Search(ctx context.Context, dense []float32, sp sparse.Vector, k int) ([]domain.Chunk, error) {
	ctx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	filter := &q.Filter{Must: []*q.Condition{q.NewMatch("schema", r.schema)}}
	req := &q.QueryPoints{CollectionName: r.collection, Limit: q.PtrOf(uint64(k)), WithPayload: q.NewWithPayload(true), Filter: filter}
	if len(sp.Indices) == 0 {
		req.Query = q.NewQueryDense(dense)
		req.Using = q.PtrOf("dense")
	} else {
		req.Prefetch = []*q.PrefetchQuery{{Query: q.NewQueryDense(dense), Using: q.PtrOf("dense"), Limit: q.PtrOf(uint64(k)), Filter: filter}, {Query: q.NewQuerySparse(sp.Indices, sp.Values), Using: q.PtrOf("sparse"), Limit: q.PtrOf(uint64(k)), Filter: filter}}
		req.Query = q.NewQueryFusion(q.Fusion_RRF)
	}
	points, err := r.client.Query(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("hybrid search: %w", err)
	}
	out := make([]domain.Chunk, 0, len(points))
	for _, p := range points {
		c := decodeChunk(p.Id, p.Payload)
		c.Score = p.Score
		out = append(out, c)
	}
	return out, nil
}
func value(v *q.Value) any {
	if v == nil {
		return nil
	}
	switch x := v.Kind.(type) {
	case *q.Value_StringValue:
		return x.StringValue
	case *q.Value_IntegerValue:
		return x.IntegerValue
	case *q.Value_DoubleValue:
		return x.DoubleValue
	case *q.Value_BoolValue:
		return x.BoolValue
	case *q.Value_StructValue:
		m := map[string]any{}
		for k, v := range x.StructValue.Fields {
			m[k] = value(v)
		}
		return m
	case *q.Value_ListValue:
		a := []any{}
		for _, v := range x.ListValue.Values {
			a = append(a, value(v))
		}
		return a
	}
	return nil
}
func decodeChunk(id *q.PointId, p map[string]*q.Value) domain.Chunk {
	s := id.GetUuid()
	if s == "" {
		s = strconv.FormatUint(id.GetNum(), 10)
	}
	m, _ := value(p["metadata"]).(map[string]any)
	return domain.Chunk{ID: s, Text: p["page_content"].GetStringValue(), Metadata: m}
}
func (r *Qdrant) Info(ctx context.Context) (*q.CollectionInfo, error) {
	ctx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	return r.client.GetCollectionInfo(ctx, r.collection)
}
func (r *Qdrant) Get(ctx context.Context, id string) ([]domain.Chunk, error) {
	ctx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	p, err := r.client.Get(ctx, &q.GetPoints{CollectionName: r.collection, Ids: []*q.PointId{q.NewIDUUID(id)}, WithPayload: q.NewWithPayload(true)})
	if err != nil {
		return nil, err
	}
	out := []domain.Chunk{}
	for _, v := range p {
		out = append(out, decodeChunk(v.Id, v.Payload))
	}
	return out, nil
}
