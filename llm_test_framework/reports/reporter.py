"""Simple JSON + HTML reporting for test runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Template

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><title>LLM Test Report</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem; }
  table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
  th, td { border: 1px solid #ddd; padding: 0.5rem; text-align: left; }
  th { background: #f5f5f5; }
  .pass { color: green; } .fail { color: red; }
  h1 { margin-bottom: 0; }
  .meta { color: #666; margin-bottom: 1rem; }
</style>
</head>
<body>
<h1>LLM Test Report</h1>
<p class="meta">Generated: {{ report.timestamp }} | Provider: {{ report.provider }}
    | Model: {{ report.model }}</p>
<table>
<tr><th>Test</th><th>Result</th><th>Score</th><th>Latency (ms)</th><th>Details</th></tr>
{% for r in report.results %}
<tr>
  <td>{{ r.test_name }}</td>
  <td class="{{ 'pass' if r.passed else 'fail' }}">{{ "PASS" if r.passed else "FAIL" }}</td>
  <td>{{ "%.3f"|format(r.score) if r.score is not none else "—" }}</td>
  <td>{{ "%.1f"|format(r.latency_ms) if r.latency_ms is not none else "—" }}</td>
  <td>{{ r.details }}</td>
</tr>
{% endfor %}
</table>
</body></html>
"""


@dataclass
class TestResultEntry:
    test_name: str
    passed: bool
    score: float | None = None
    latency_ms: float | None = None
    details: str = ""


@dataclass
class TestReport:
    provider: str
    model: str
    results: list[TestResultEntry] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add(self, entry: TestResultEntry) -> None:
        self.results.append(entry)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)


def generate_report(report: TestReport, output_dir: str | Path = "reports") -> Path:
    """Write JSON and HTML reports. Returns path to the HTML file."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "report.json"
    json_path.write_text(json.dumps(asdict(report), indent=2, default=str))

    html_path = out / "report.html"
    html = Template(_HTML_TEMPLATE).render(report=report)
    html_path.write_text(html)

    return html_path
