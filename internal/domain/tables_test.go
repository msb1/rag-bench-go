package domain

import (
	"strings"
	"testing"
)

func TestCleanTablePreservesEmptyCellsAndData(t *testing.T) {
	s := "| Name | Value |\n| --- | --- |\n| A | |\ncontinued from previous page\n| Name | Value |\n| --- | --- |\n| B | &amp;lt;4<br/>units |\n| A | |"
	out := CleanTable(s)
	if strings.Count(out, "| Name | Value |") != 1 || strings.Count(out, "| --- | --- |") != 1 || strings.Count(out, "| A | |") != 2 || !strings.Contains(out, "<4 units") {
		t.Fatalf("bad cleanup:\n%s", out)
	}
}
