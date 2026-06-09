package main

import (
	"flag"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/openfluke/planetbridging/host"
)

func main() {
	cmd := "host"
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "host":
			cmd = "host"
			os.Args = append([]string{os.Args[0]}, os.Args[2:]...)
		default:
			log.Fatalf("unknown command %q (usage: go run . [host])", os.Args[1])
		}
	}

	switch cmd {
	case "host":
		runHost()
	default:
		log.Fatalf("unknown command %q", cmd)
	}
}

func runHost() {
	addr := flag.String("addr", ":9876", "listen address")
	denseReports := flag.String("reports", defaultDenseReportsDir(), "dense JSON reports dir")
	cnn1Reports := flag.String("cnn1-reports", defaultCNN1ReportsDir(), "cnn1 JSON reports dir")
	flag.Parse()

	denseStore, err := host.NewStore(*denseReports)
	if err != nil {
		log.Fatal(err)
	}
	cnn1Store, err := host.NewStore(*cnn1Reports)
	if err != nil {
		log.Fatal(err)
	}

	srv := host.NewServer(denseStore, cnn1Store, defaultDenseModelsDir(), defaultCNN1ModelsDir())
	log.Printf("planetbridging host listening on %s", *addr)
	log.Printf("Compare UI → http://localhost%s/", *addr)
	log.Printf("Compare JSON → http://localhost%s/api/v1/compare", *addr)
	if err := http.ListenAndServe(*addr, srv.Handler()); err != nil {
		log.Fatal(err)
	}
}

func defaultDenseReportsDir() string {
	return bedrockPath("dense", "reports")
}

func defaultCNN1ReportsDir() string {
	return bedrockPath("cnn1", "reports")
}

func defaultDenseModelsDir() string {
	return bedrockPath("dense", "models")
}

func defaultCNN1ModelsDir() string {
	return bedrockPath("cnn1", "models")
}

func bedrockPath(bedrock, sub string) string {
	if wd, err := os.Getwd(); err == nil {
		candidate := filepath.Join(wd, "python", bedrock, sub)
		if _, err := os.Stat(filepath.Join(wd, "python", bedrock)); err == nil {
			return candidate
		}
	}
	return filepath.Join("python", bedrock, sub)
}
