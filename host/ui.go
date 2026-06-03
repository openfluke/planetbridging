package host

import (
	"bytes"
	"fmt"
	"html/template"
	"net/http"
)

func (s *Server) handleIndex(w http.ResponseWriter, _ *http.Request) {
	dash, err := s.buildDashboard()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	var buf bytes.Buffer
	if err := dashboardTmpl.Execute(&buf, dash); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write(buf.Bytes())
}

var dashboardTmpl = template.Must(template.New("dashboard").Funcs(template.FuncMap{
	"exactClass": func(exact bool, pending bool) string {
		if pending {
			return "pending"
		}
		if exact {
			return "exact"
		}
		return "diff"
	},
	"exactLabel": func(exact bool, pending bool) string {
		if pending {
			return "PENDING"
		}
		if exact {
			return "EXACT"
		}
		return "DIFF"
	},
	"fmtSci": func(v float64) string {
		return fmt.Sprintf("%.6e", v)
	},
	"stepLabel": func(stage, format string) string {
		if stage == format {
			return stage
		}
		return stage + " / " + format
	},
}).Parse(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Planet Bridging — Dense</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0d0d0d;
    --panel: #141414;
    --border: #2a2a2a;
    --text: #d4d4d4;
    --muted: #888;
    --accent: #7ec8e3;
    --loom: #ffb74d;
    --exact: #4caf50;
    --diff: #ffb74d;
    --pending: #666;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 13px;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }
  header {
    padding: 24px 32px;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
  }
  h1 { margin: 0 0 8px; font-size: 18px; color: var(--accent); font-weight: 600; }
  .meta { color: var(--muted); font-size: 12px; margin-bottom: 12px; }
  .fixture-banner {
    padding: 10px 14px;
    border: 1px solid #1e3a4a;
    border-radius: 6px;
    background: rgba(126, 200, 227, 0.06);
    font-size: 12px;
  }
  .pipeline-hint {
    margin-top: 10px;
    color: var(--muted);
    font-size: 12px;
  }
  main { padding: 24px 32px 48px; max-width: 1400px; }
  .empty {
    border: 1px dashed var(--border);
    padding: 32px;
    color: var(--muted);
    border-radius: 8px;
  }
  .empty code { color: var(--accent); }
  section.model {
    margin-bottom: 28px;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    background: var(--panel);
  }
  section.model h2 {
    margin: 0;
    padding: 14px 18px;
    font-size: 14px;
    border-bottom: 1px solid var(--border);
    background: #101010;
  }
  .planet-block {
    border-top: 1px solid var(--border);
  }
  .planet-block:first-child { border-top: none; }
  .planet-head {
    padding: 10px 18px;
    background: #111;
    color: var(--accent);
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .steps {
    padding: 8px 18px 4px;
    color: var(--muted);
    font-size: 11px;
  }
  .steps span {
    display: inline-block;
    margin-right: 12px;
    color: var(--text);
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  th, td {
    padding: 8px 14px;
    text-align: left;
    border-top: 1px solid var(--border);
  }
  th { color: var(--muted); font-weight: 500; }
  tr.exact td { background: rgba(76, 175, 80, 0.08); }
  tr.diff td { background: rgba(255, 183, 77, 0.05); }
  tr.pending td { background: rgba(102, 102, 102, 0.08); color: var(--muted); }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }
  .badge.exact { background: rgba(76,175,80,.2); color: var(--exact); }
  .badge.diff { background: rgba(255,183,77,.15); color: var(--diff); }
  .badge.pending { background: rgba(102,102,102,.2); color: #aaa; }
  footer {
    padding: 16px 32px;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 11px;
  }
  a { color: var(--accent); }
</style>
</head>
<body>
<header>
  <h1>Dense — planet pipeline compare</h1>
  <div class="meta">
    fixture <strong>{{.Fixture.Version}}</strong>
    · seed <strong>{{.Fixture.Seed}}</strong>
    · {{.Dense.ReportCount}} pipeline reports
  </div>
  <div class="fixture-banner">
    <strong>Same training data per planet.</strong> {{.Fixture.Note}}
  </div>
  <div class="pipeline-hint">
    Per planet: <strong>native</strong> (train + infer) → <strong>export</strong> (reload saved format) → <strong>loom</strong> (import + infer, coming soon).
    We do <em>not</em> cross-compare TensorFlow vs PyTorch — only steps within each planet's pipeline.
  </div>
</header>

<main>
{{if not .Dense.Models}}
  <div class="empty">
    No pipeline reports yet.<br><br>
    <code>go run .</code><br>
    <code>./python/dense/run_dense.sh</code>
  </div>
{{else}}
  {{range .Dense.Models}}
  <section class="model">
    <h2>{{.ModelID}}</h2>
    {{range .Pipelines}}
    <div class="planet-block">
      <div class="planet-head">{{.Planet}} pipeline</div>
      <div class="steps">
        {{range .Steps}}
        <span>{{stepLabel .Stage .Format}} ✓</span>
        {{end}}
      </div>
      <table>
        <thead>
          <tr>
            <th>Compare</th>
            <th>From</th>
            <th>To</th>
            <th>Max abs diff</th>
            <th>Mean abs diff</th>
          </tr>
        </thead>
        <tbody>
          {{range .Compare}}
          <tr class="{{exactClass .ExactMatch .Pending}}">
            <td><span class="badge {{exactClass .ExactMatch .Pending}}">{{exactLabel .ExactMatch .Pending}}</span></td>
            <td>{{.FromStage}} / {{.FromFormat}}</td>
            <td>{{if .Pending}}loom (not imported yet){{else}}{{.ToStage}} / {{.ToFormat}}{{end}}</td>
            <td>{{if .Pending}}—{{else}}{{fmtSci .MaxAbsDiff}}{{end}}</td>
            <td>{{if .Pending}}—{{else}}{{fmtSci .MeanAbsDiff}}{{end}}</td>
          </tr>
          {{end}}
        </tbody>
      </table>
    </div>
    {{end}}
  </section>
  {{end}}
{{end}}
</main>

<footer>
  JSON: <a href="/api/v1/compare">/api/v1/compare</a>
  · saved models: <a href="/api/v1/loom/catalog">/api/v1/loom/catalog</a>
</footer>
</body>
</html>`))
