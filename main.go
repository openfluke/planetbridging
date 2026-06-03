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
	reportsDir := flag.String("reports", defaultReportsDir(), "directory for JSON reports")
	flag.Parse()

	store, err := host.NewStore(*reportsDir)
	if err != nil {
		log.Fatal(err)
	}

	srv := host.NewServer(store, defaultModelsDir())
	log.Printf("planetbridging host listening on %s", *addr)
	log.Printf("Compare UI → http://localhost%s/", *addr)
	log.Printf("Compare JSON → http://localhost%s/api/v1/compare", *addr)
	if err := http.ListenAndServe(*addr, srv.Handler()); err != nil {
		log.Fatal(err)
	}
}

func defaultReportsDir() string {
	if wd, err := os.Getwd(); err == nil {
		candidate := filepath.Join(wd, "python", "dense", "reports")
		if _, err := os.Stat(filepath.Join(wd, "python", "dense")); err == nil {
			return candidate
		}
	}
	return "python/dense/reports"
}

func defaultModelsDir() string {
	if wd, err := os.Getwd(); err == nil {
		candidate := filepath.Join(wd, "python", "dense", "models")
		if _, err := os.Stat(filepath.Join(wd, "python", "dense")); err == nil {
			return candidate
		}
	}
	return "python/dense/models"
}
