package domain

import "strings"

type Chunk struct {
	ID       string         `json:"id"`
	Text     string         `json:"page_content"`
	Metadata map[string]any `json:"metadata"`
	Score    float32        `json:"score,omitempty"`
}

func (c Chunk) Context() string {
	text := c.Text
	var tables []string
	switch v := c.Metadata["raw_table_content"].(type) {
	case []string:
		tables = v
	case []any:
		for _, t := range v {
			if s, ok := t.(string); ok {
				tables = append(tables, s)
			}
		}
	}
	if len(tables) > 0 {
		cleaned := make([]string, len(tables))
		for i, table := range tables {
			cleaned[i] = CleanTable(table)
		}
		text += "\n\n### Associated Table Data\n" + strings.Join(cleaned, "\n\n")
	}
	return text
}

type Record struct {
	QueryID     string   `json:"query_id,omitempty"`
	Question    string   `json:"question"`
	Answer      string   `json:"answer"`
	Contexts    []string `json:"contexts"`
	GroundTruth string   `json:"ground_truth"`
}
type Metrics struct {
	Faithfulness      float64 `json:"faithfulness"`
	AnswerRelevancy   float64 `json:"answer_relevancy"`
	ContextRecall     float64 `json:"context_recall"`
	ContextPrecision  float64 `json:"context_precision"`
	AnswerCorrectness float64 `json:"answer_correctness"`
}

func (m Metrics) Values() map[string]float64 {
	return map[string]float64{"faithfulness": m.Faithfulness, "answer_relevancy": m.AnswerRelevancy, "context_recall": m.ContextRecall, "context_precision": m.ContextPrecision, "answer_correctness": m.AnswerCorrectness}
}

type Evaluated struct {
	Record
	Metrics
}
type RAGResult struct {
	Record
	Queries  []string `json:"queries"`
	Sources  []Chunk  `json:"sources"`
	Warnings []string `json:"warnings,omitempty"`
}
