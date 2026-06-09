package host

import (
	"bytes"
	"strings"

	"github.com/go-pdf/fpdf"
)

const pdfLineHeight = 3.6

// sanitizePDFText maps Unicode punctuation to ASCII for fpdf core fonts (Latin-1 only).
func sanitizePDFText(s string) string {
	return strings.NewReplacer(
		"—", "-", // em dash
		"–", "-", // en dash
		"≈", "~", // approximately
		"→", "->",
		"≥", ">=",
		"·", ",",
	).Replace(s)
}

// RenderAllComparisonPDF renders the full multi-bedrock compare log as a PDF.
func RenderAllComparisonPDF(version string, summaries map[string]DenseComparisonSummary) ([]byte, error) {
	text := FormatAllComparisonText(version, summaries)

	pdf := fpdf.New("P", "mm", "A4", "")
	pdf.SetMargins(12, 14, 12)
	pdf.SetAutoPageBreak(true, 14)
	pdf.AddPage()
	pdf.SetFont("Courier", "", 7)

	pageW, _ := pdf.GetPageSize()
	lm, _, rm, _ := pdf.GetMargins()
	width := pageW - lm - rm

	for _, line := range strings.Split(text, "\n") {
		pdf.MultiCell(width, pdfLineHeight, sanitizePDFText(line), "", "L", false)
	}

	var buf bytes.Buffer
	if err := pdf.Output(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
