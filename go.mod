module github.com/openfluke/planetbridging

go 1.26.2

require (
	github.com/go-pdf/fpdf v0.9.0
	github.com/openfluke/loom v0.0.0
)

require github.com/openfluke/webgpu v1.0.4 // indirect

replace github.com/openfluke/loom => ../
