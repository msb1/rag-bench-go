// Package chunker implements header-aware, recursive token-bounded Markdown splitting.
package chunker

import (
	"fmt"
	"regexp"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/google/uuid"
	"github.com/tiktoken-go/tokenizer"
	"rag-bench-go/internal/domain"
)

type Chunker struct {
	size, overlap int
	codec         tokenizer.Codec
	mu            sync.Mutex
}

func New(size, overlap int) (*Chunker, error) {
	if size < 1 || overlap < 0 || overlap >= size {
		return nil, fmt.Errorf("invalid chunk size/overlap")
	}
	c, err := tokenizer.Get(tokenizer.Cl100kBase)
	if err != nil {
		return nil, err
	}
	return &Chunker{size: size, overlap: overlap, codec: c}, nil
}

var header = regexp.MustCompile(`^(#{1,6})\s+(.+?)\s*#*\s*$`)
var reference = regexp.MustCompile(`(?i)^(references|bibliography|works\s+cited)\b`)
var tableSeparator = regexp.MustCompile(`^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$`)
var math = regexp.MustCompile(`(?s)\$\$.*?\$\$|\$[^\n$]*\$|\\[a-zA-Z]+`)
var words = regexp.MustCompile(`\b[a-zA-Z]{2,}\b`)

type section struct {
	text    string
	headers map[string]any
	tables  []string
}

func sections(markdown string) []section {
	lines := strings.Split(strings.ReplaceAll(markdown, "\r\n", "\n"), "\n")
	var out []section
	var body []string
	var tables []string
	heads := map[string]any{}
	fence := ""
	flush := func() {
		if len(body) > 0 || len(tables) > 0 {
			h := map[string]any{}
			for k, v := range heads {
				h[k] = v
			}
			out = append(out, section{strings.TrimSpace(strings.Join(body, "\n")), h, tables})
		}
		body = nil
		tables = nil
	}
	for i := 0; i < len(lines); i++ {
		line := lines[i]
		trim := strings.TrimSpace(line)
		if strings.HasPrefix(trim, "```") || strings.HasPrefix(trim, "~~~") {
			mark := trim[:3]
			if fence == "" {
				fence = mark
			} else if fence == mark {
				fence = ""
			}
			body = append(body, line)
			continue
		}
		if fence != "" {
			body = append(body, line)
			continue
		}
		if m := header.FindStringSubmatch(line); m != nil {
			if reference.MatchString(m[2]) {
				break
			}
			flush()
			level := len(m[1])
			for j := level; j <= 6; j++ {
				delete(heads, fmt.Sprintf("Header %d", j))
			}
			heads[fmt.Sprintf("Header %d", level)] = m[2]
		}
		if i+1 < len(lines) && strings.Contains(line, "|") && tableSeparator.MatchString(lines[i+1]) {
			// Anchor tables to the preceding prose instead of allowing placeholders to split.
			table := []string{line, lines[i+1]}
			i += 2
			for i < len(lines) && strings.Contains(lines[i], "|") && strings.TrimSpace(lines[i]) != "" {
				table = append(table, lines[i])
				i++
			}
			i--
			tables = append(tables, strings.Join(table, "\n"))
			flush()
			continue
		}
		body = append(body, line)
	}
	flush()
	return out
}

func (c *Chunker) Count(text string) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.codec.Count(text)
}

// split prefers paragraphs, lines and words, with a UTF-8-safe token fallback.
func (c *Chunker) split(text string) ([]string, error) {
	var out []string
	for strings.TrimSpace(text) != "" {
		text = strings.TrimSpace(text)
		ids, pieces, err := c.codec.Encode(text)
		if err != nil {
			return nil, err
		}
		if len(ids) <= c.size {
			out = append(out, text)
			break
		}
		end := 0
		for _, p := range pieces[:c.size] {
			end += len(p)
		}
		for end > 0 && !utf8.ValidString(text[:end]) {
			end--
		}
		if end == 0 {
			return nil, fmt.Errorf("token budget cannot hold a Unicode character")
		}
		// Boundaries in the latter half keep chunks useful while respecting structure.
		for _, sep := range []string{"\n\n", "\n", ". ", " "} {
			if p := strings.LastIndex(text[:end], sep); p >= end/2 {
				end = p + len(sep)
				break
			}
		}
		part := strings.TrimSpace(text[:end])
		// Re-tokenizing a boundary can change BPE merges; enforce the actual limit.
		for {
			n, e := c.codec.Count(part)
			if e != nil {
				return nil, e
			}
			if n <= c.size {
				break
			}
			_, w := utf8.DecodeLastRuneInString(part)
			part = part[:len(part)-w]
			end = len(part)
		}
		out = append(out, part)
		_, tokens, e := c.codec.Encode(text[:end])
		if e != nil {
			return nil, e
		}
		over := 0
		if c.overlap > 0 {
			for _, p := range tokens[max(0, len(tokens)-c.overlap):] {
				over += len(p)
			}
		}
		start := end - over
		for start < end && !utf8.RuneStart(text[start]) {
			start++
		}
		if start <= 0 {
			start = end
		}
		text = text[start:]
	}
	return out, nil
}

func noise(text string) bool {
	return len(text) < 60 || len(words.FindAllString(math.ReplaceAllString(text, " "), -1)) < 10 || (strings.HasPrefix(text, "![") && strings.HasSuffix(text, ")"))
}

func (c *Chunker) Chunk(markdown, key string) ([]domain.Chunk, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !utf8.ValidString(markdown) {
		return nil, fmt.Errorf("Markdown must be valid UTF-8")
	}
	out := []domain.Chunk{}
	for _, s := range sections(markdown) {
		parts, err := c.split(s.text)
		if err != nil {
			return nil, err
		}
		if len(parts) == 0 && len(s.tables) > 0 {
			parts = []string{""}
		}
		for i, part := range parts {
			hasTables := i == len(parts)-1 && len(s.tables) > 0
			if noise(part) && !hasTables {
				continue
			}
			metadata := map[string]any{"type": "text", "source": key, "tokenizer": "cl100k_base"}
			for k, v := range s.headers {
				metadata[k] = v
			}
			if hasTables {
				metadata["raw_table_content"] = s.tables
				metadata["has_tables"] = true
			}
			n, err := c.codec.Count(part)
			if err != nil {
				return nil, err
			}
			metadata["token_count"] = n
			out = append(out, domain.Chunk{Text: part, Metadata: metadata})
		}
	}
	for i := range out {
		logical := fmt.Sprintf("%s__chunk__%d__%d", key, i+1, len(out))
		out[i].ID = uuid.NewSHA1(uuid.NameSpaceDNS, []byte(logical)).String()
		out[i].Metadata["logical_chunk_id"] = logical
	}
	return out, nil
}
