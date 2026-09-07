package domain

import (
	"html"
	"regexp"
	"strings"
)

var breaks = regexp.MustCompile(`(?i)<br\s*/?>`)
var separator = regexp.MustCompile(`^\|?[:\-\s|]+\|?$`)

// CleanTable removes repeated continuation headers while preserving data rows
// and empty cells. Raw Markdown is still available in the chunk's metadata.
func CleanTable(table string) string {
	table = html.UnescapeString(html.UnescapeString(breaks.ReplaceAllString(table, " ")))
	out := []string{}
	header := ""
	seenSeparator := false
	for _, line := range strings.Split(table, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.Contains(strings.ToLower(line), "continued from previous page") {
			continue
		}
		normalized := strings.ToLower(strings.Join(strings.Fields(line), ""))
		if strings.Contains(line, "|") {
			if separator.MatchString(line) {
				if seenSeparator {
					continue
				}
				seenSeparator = true
			} else if header == "" {
				header = normalized
			} else if normalized == header {
				continue
			}
		}
		out = append(out, line)
	}
	return strings.Join(out, "\n")
}
