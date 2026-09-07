package repository

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// ReadJSONL streams records and reports malformed lines instead of dropping data.
func ReadJSONL[T any](path string, consume func(T) error) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 64<<10), 32<<20)
	line := 0
	for s.Scan() {
		line++
		data := bytes.TrimSpace(s.Bytes())
		if len(data) == 0 {
			continue
		}
		// Accept Python benchmark rows whose contexts accidentally contain one string.
		var raw map[string]json.RawMessage
		if err = json.Unmarshal(data, &raw); err != nil {
			return fmt.Errorf("%s line %d: %w", path, line, err)
		}
		if c := raw["contexts"]; len(c) > 0 && c[0] == '"' {
			var text string
			if err = json.Unmarshal(c, &text); err != nil {
				return err
			}
			raw["contexts"], err = json.Marshal([]string{text})
			if err != nil {
				return err
			}
			data, err = json.Marshal(raw)
			if err != nil {
				return err
			}
		}
		var v T
		if err = json.Unmarshal(data, &v); err != nil {
			return fmt.Errorf("%s line %d: %w", path, line, err)
		}
		if err = consume(v); err != nil {
			return fmt.Errorf("%s line %d: %w", path, line, err)
		}
	}
	return s.Err()
}
func AppendJSONL(path string, v any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return err
	}
	defer f.Close()
	stat, err := f.Stat()
	if err != nil {
		return err
	}
	if stat.Size() > 0 {
		var b [1]byte
		if _, err = f.ReadAt(b[:], stat.Size()-1); err != nil {
			return err
		}
		if b[0] != '\n' {
			return fmt.Errorf("%s has an incomplete final line; repair it before resuming", path)
		}
	}
	if _, err = f.Seek(0, io.SeekEnd); err != nil {
		return err
	}
	if err = json.NewEncoder(f).Encode(v); err != nil {
		return err
	}
	return f.Sync()
}
func AtomicWrite(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".rag-bench-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	if _, err = f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err = f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}
func WriteJSONL[T any](path string, rows []T) error {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	for _, r := range rows {
		if err := enc.Encode(r); err != nil {
			return err
		}
	}
	return AtomicWrite(path, buf.Bytes())
}
