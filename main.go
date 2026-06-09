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
	cnn2Reports := flag.String("cnn2-reports", defaultCNN2ReportsDir(), "cnn2 JSON reports dir")
	cnn3Reports := flag.String("cnn3-reports", defaultCNN3ReportsDir(), "cnn3 JSON reports dir")
	mhaReports := flag.String("mha-reports", defaultMHAReportsDir(), "mha JSON reports dir")
	lstmReports := flag.String("lstm-reports", defaultLSTMReportsDir(), "lstm JSON reports dir")
	rnnReports := flag.String("rnn-reports", defaultRNNReportsDir(), "rnn JSON reports dir")
	mixerReports := flag.String("mixer-reports", defaultMixerReportsDir(), "mixer JSON reports dir")
	flag.Parse()

	denseStore, err := host.NewStore(*denseReports)
	if err != nil {
		log.Fatal(err)
	}
	cnn1Store, err := host.NewStore(*cnn1Reports)
	if err != nil {
		log.Fatal(err)
	}
	cnn2Store, err := host.NewStore(*cnn2Reports)
	if err != nil {
		log.Fatal(err)
	}
	cnn3Store, err := host.NewStore(*cnn3Reports)
	if err != nil {
		log.Fatal(err)
	}
	mhaStore, err := host.NewStore(*mhaReports)
	if err != nil {
		log.Fatal(err)
	}
	lstmStore, err := host.NewStore(*lstmReports)
	if err != nil {
		log.Fatal(err)
	}
	rnnStore, err := host.NewStore(*rnnReports)
	if err != nil {
		log.Fatal(err)
	}
	mixerStore, err := host.NewStore(*mixerReports)
	if err != nil {
		log.Fatal(err)
	}

	srv := host.NewServer(
		denseStore, cnn1Store, cnn2Store, cnn3Store, mhaStore, lstmStore, rnnStore, mixerStore,
		defaultDenseModelsDir(), defaultCNN1ModelsDir(), defaultCNN2ModelsDir(), defaultCNN3ModelsDir(), defaultMHAModelsDir(), defaultLSTMModelsDir(), defaultRNNModelsDir(), defaultMixerModelsDir(),
	)
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

func defaultCNN2ReportsDir() string {
	return bedrockPath("cnn2", "reports")
}

func defaultCNN3ReportsDir() string {
	return bedrockPath("cnn3", "reports")
}

func defaultMHAReportsDir() string {
	return bedrockPath("mha", "reports")
}

func defaultLSTMReportsDir() string {
	return bedrockPath("lstm", "reports")
}

func defaultRNNReportsDir() string {
	return bedrockPath("rnn", "reports")
}

func defaultMixerReportsDir() string {
	return bedrockPath("mixer", "reports")
}

func defaultDenseModelsDir() string {
	return bedrockPath("dense", "models")
}

func defaultCNN1ModelsDir() string {
	return bedrockPath("cnn1", "models")
}

func defaultCNN2ModelsDir() string {
	return bedrockPath("cnn2", "models")
}

func defaultCNN3ModelsDir() string {
	return bedrockPath("cnn3", "models")
}

func defaultMHAModelsDir() string {
	return bedrockPath("mha", "models")
}

func defaultLSTMModelsDir() string {
	return bedrockPath("lstm", "models")
}

func defaultRNNModelsDir() string {
	return bedrockPath("rnn", "models")
}

func defaultMixerModelsDir() string {
	return bedrockPath("mixer", "models")
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
