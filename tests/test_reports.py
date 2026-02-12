"""Unit tests for the reporting module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_test_framework.reports import TestReport, TestResultEntry, generate_report


class TestTestReport:
    def test_pass_rate(self):
        report = TestReport(provider="mock", model="m")
        report.add(TestResultEntry("t1", passed=True))
        report.add(TestResultEntry("t2", passed=False))
        report.add(TestResultEntry("t3", passed=True))
        assert report.pass_rate == pytest.approx(2 / 3)

    def test_empty_report(self):
        report = TestReport(provider="mock", model="m")
        assert report.pass_rate == 0.0


class TestGenerateReport:
    def test_creates_files(self, tmp_path: Path):
        report = TestReport(provider="mock", model="mock-model")
        report.add(TestResultEntry("test_1", passed=True, score=0.95, latency_ms=12.3))
        report.add(TestResultEntry("test_2", passed=False, details="Below threshold"))

        html_path = generate_report(report, output_dir=tmp_path)

        assert html_path.exists()
        assert (tmp_path / "report.json").exists()

        data = json.loads((tmp_path / "report.json").read_text())
        assert data["provider"] == "mock"
        assert len(data["results"]) == 2

        html = html_path.read_text()
        assert "test_1" in html
        assert "FAIL" in html
