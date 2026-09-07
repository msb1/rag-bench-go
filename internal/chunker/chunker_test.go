package chunker

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestTokenLimitStructureAndTables(t *testing.T) {
	c, err := New(500, 50)
	if err != nil {
		t.Fatal(err)
	}
	prose := strings.Repeat("These detailed scientific results explain how the retrieval system works across multiple documents. ", 160)
	table := "| Model | Accuracy |\n| --- | --- |\n| Gemma | 0.95 |"
	md := "# Research\n## Results\n" + prose + "\n\n" + table + "\n\n## References\nThis reference section must never reach the index."
	chunks, err := c.Chunk(md, "markdown/paper.md")
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) < 3 {
		t.Fatalf("expected multiple token-sized chunks, got %d", len(chunks))
	}
	tables := 0
	large := false
	for _, ch := range chunks {
		n, err := c.Count(ch.Text)
		if err != nil || n > 500 {
			t.Fatalf("token limit: %d %v", n, err)
		}
		if len(ch.Text) > 500 {
			large = true
		}
		if !utf8.ValidString(ch.Text) {
			t.Fatal("invalid UTF-8")
		}
		if strings.Contains(ch.Text, "must never") {
			t.Fatal("references retained")
		}
		if ch.Metadata["Header 1"] != "Research" || ch.Metadata["Header 2"] != "Results" {
			t.Fatalf("headers lost: %v", ch.Metadata)
		}
		if strings.Contains(ch.Context(), table) {
			tables++
		}
	}
	if !large {
		t.Fatal("still splitting at 500 characters")
	}
	if tables != 1 {
		t.Fatalf("table attached %d times", tables)
	}
	again, _ := c.Chunk(md, "markdown/paper.md")
	for i := range chunks {
		if chunks[i].ID != again[i].ID {
			t.Fatal("unstable IDs")
		}
	}
}
func TestTableOnlyAndFencedReferences(t *testing.T) {
	c, _ := New(500, 50)
	table := "| A | B |\n| :--- | ---: |\n| 1 | 2 |"
	chunks, err := c.Chunk(table, "table.md")
	if err != nil || len(chunks) != 1 || !strings.Contains(chunks[0].Context(), table) {
		t.Fatalf("table-only document lost: %+v %v", chunks, err)
	}
	md := "```markdown\n# References\nThese are useful words inside a code example and should survive the references filter.\n```\n"
	chunks, err = c.Chunk(md, "code.md")
	if err != nil || len(chunks) == 0 || !strings.Contains(chunks[0].Text, "# References") {
		t.Fatal("fenced header incorrectly stripped")
	}
}
func TestUnicodeSplitterPreservesContent(t *testing.T) {
	c, _ := New(17, 0)
	original := strings.Repeat("😀日本語café", 40)
	c.mu.Lock()
	parts, err := c.split(original)
	c.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(parts, "") != original {
		t.Fatal("Unicode content lost or duplicated")
	}
	for _, p := range parts {
		n, _ := c.Count(p)
		if !utf8.ValidString(p) || n > 17 {
			t.Fatalf("bad chunk: %q (%d)", p, n)
		}
	}
}
func TestNoiseAndHeaderHierarchy(t *testing.T) {
	c, _ := New(500, 50)
	chunks, err := c.Chunk("# Intro\n$\\alpha$\n\n![image](x.png)\n12\n", "noise.md")
	if err != nil || len(chunks) != 0 {
		t.Fatalf("noise indexed: %v %v", chunks, err)
	}
	text := "These are more than ten ordinary English words explaining the current section in detail."
	chunks, _ = c.Chunk("# One\n## Child\n"+text+"\n# Two\n"+text, "hierarchy.md")
	if len(chunks) != 2 {
		t.Fatalf("%+v", chunks)
	}
	if _, ok := chunks[1].Metadata["Header 2"]; ok {
		t.Fatal("stale child heading")
	}
}
