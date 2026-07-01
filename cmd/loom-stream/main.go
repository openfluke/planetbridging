// loom-stream reads a layer-stream JSON envelope from stdin and writes a .entity
// checkpoint plus Loom inference outputs to stdout — no HTTP compare-host needed.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/openfluke/planetbridging/bridge"
)

func main() {
	bedrock := flag.String("bedrock", "", "bedrock type: dense, cnn1, mha, … (auto-detect if omitted)")
	root := flag.String("root", "", "planetbridging repo root (default fixtures/models under python/<bedrock>/)")
	modelsDir := flag.String("models-dir", "", "where to write .entity files")
	fixturesDir := flag.String("fixtures-dir", "", "bedrock fixture npz directory")
	fixtureVersion := flag.String("fixture-version", "", "fixture version tag (e.g. dense_bedrock_v2)")
	outputPath := flag.String("output", "", "explicit .entity output path (overrides models-dir layout)")
	skipInfer := flag.Bool("skip-infer", false, "only write .entity, skip Loom forward pass")
	flag.Parse()

	body, err := io.ReadAll(os.Stdin)
	if err != nil {
		fatal(err)
	}
	if len(body) == 0 {
		fatal(fmt.Errorf("stdin JSON required"))
	}

	// Accept either a CLI envelope or a bare stream payload.
	var envelope bridge.CLIRequest
	if err := json.Unmarshal(body, &envelope); err != nil {
		fatal(err)
	}
	if envelope.Bedrock == "" && *bedrock != "" {
		envelope.Bedrock = *bedrock
	}
	if envelope.Root == "" && *root != "" {
		envelope.Root = *root
	}
	if envelope.ModelsDir == "" && *modelsDir != "" {
		envelope.ModelsDir = *modelsDir
	}
	if envelope.FixturesDir == "" && *fixturesDir != "" {
		envelope.FixturesDir = *fixturesDir
	}
	if envelope.FixtureVersion == "" && *fixtureVersion != "" {
		envelope.FixtureVersion = *fixtureVersion
	}
	if envelope.OutputPath == "" && *outputPath != "" {
		envelope.OutputPath = *outputPath
	}
	if !envelope.SkipInfer {
		envelope.SkipInfer = *skipInfer
	}

	payload := body
	if len(envelope.Payload) > 0 {
		payload = envelope.Payload
	}

	resp := bridge.StreamFromCLI(envelope, payload)
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(resp); err != nil {
		fatal(err)
	}
	if resp.Status != "ok" {
		os.Exit(1)
	}
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "loom-stream: %v\n", err)
	os.Exit(1)
}
