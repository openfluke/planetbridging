package host

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHandleCompareLayerNormEmpty(t *testing.T) {
	empty := t.TempDir()
	mk := func() *Store {
		s, err := NewStore(empty)
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	s := NewServer(
		mk(), mk(), mk(), mk(), mk(), mk(), mk(), mk(), mk(),
		empty, empty, empty, empty, empty, empty, empty, empty, empty,
	)
	req := httptest.NewRequest(http.MethodGet, "/api/v1/compare/layernorm", nil)
	rec := httptest.NewRecorder()
	s.handleCompareLayerNorm(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d", rec.Code)
	}
	var summary DenseComparisonSummary
	if err := json.Unmarshal(rec.Body.Bytes(), &summary); err != nil {
		t.Fatalf("json: %v", err)
	}
}
