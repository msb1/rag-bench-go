package sparse

import "testing"

func TestDocumentAndQueryShareVocabulary(t *testing.T) {
	d := Document("Rare scientific observation observation observation")
	q := Query("observation rare")
	if len(q.Indices) != 2 {
		t.Fatalf("%+v", q)
	}
	for i, id := range q.Indices {
		if q.Values[i] != 1 {
			t.Fatal("query weights must be one")
		}
		found := false
		for j, did := range d.Indices {
			if did == id {
				found = true
				if d.Values[j] <= 0 {
					t.Fatal("bad BM25 weight")
				}
			}
		}
		if !found {
			t.Fatal("vocabulary mismatch")
		}
	}
	for i := 1; i < len(d.Indices); i++ {
		if d.Indices[i] <= d.Indices[i-1] {
			t.Fatal("indices not strictly sorted")
		}
	}
	if len(Query("the and of").Indices) != 0 {
		t.Fatal("stopword-only query not empty")
	}
}
