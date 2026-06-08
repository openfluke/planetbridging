package bridge_test

import (
	"path/filepath"
	"testing"

	"github.com/openfluke/planetbridging/bridge"
)

func TestFixtureRow32(t *testing.T) {
	fx, err := bridge.LoadFixture("dense_bedrock_v2", filepath.Join("..", "python", "dense", "fixtures"))
	if err != nil {
		t.Fatal(err)
	}
	if len(fx.XTest) < 40 {
		t.Fatal("short x_test")
	}
	// Values from python npz (sample 32 col 0) — catches truncated zip reads
	if fx.XTest[32][0] == fx.XTest[33][0] && fx.XTest[33][0] == fx.XTest[34][0] {
		t.Fatalf("suspicious identical rows 32-34: %v", fx.XTest[32][:4])
	}
}
