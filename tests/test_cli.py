"""Unit tests for CLI ingest behavior (network-free via monkeypatching)."""

from datetime import date
from pathlib import Path

import pytest

from rategauge import cli
from rategauge.sources import common
from rategauge.sources.common import DocumentRef

FED_REF = DocumentRef(
    "FED", "fed_20250618a", date(2025, 6, 18), "https://fed.example/a", "statement"
)
ECB_REF = DocumentRef(
    "ECB", "ecb_mp260611", date(2026, 6, 11), "https://ecb.example/b", "decision"
)


def test_filtered_ingest_preserves_other_bank_in_catalog(tmp_path, monkeypatch):
    catalog = tmp_path / "documents.csv"
    common.write_catalog([FED_REF], catalog)

    monkeypatch.setattr(cli.ecb, "enumerate_decisions", lambda **kwargs: [ECB_REF])
    monkeypatch.setattr(
        cli.common,
        "fetch_documents",
        lambda refs, cache, **kwargs: {r.doc_id: "<html/>" for r in refs},
    )
    cli.run_ingest(tmp_path / "cache", catalog, bank="ECB")

    banks = {ref.bank for ref in common.read_catalog(catalog)}
    assert banks == {"FED", "ECB"}  # the FED rows must survive an ECB-only refresh


def test_ingest_exits_nonzero_on_missing_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.fed, "enumerate_statements", lambda **kwargs: [FED_REF])
    monkeypatch.setattr(cli.ecb, "enumerate_decisions", lambda **kwargs: [ECB_REF])
    monkeypatch.setattr(cli.common, "fetch_documents", lambda refs, cache, **kwargs: {})

    with pytest.raises(SystemExit) as excinfo:
        cli.run_ingest(tmp_path / "cache", tmp_path / "documents.csv", bank="all")
    assert excinfo.value.code == 1


def test_catalog_round_trip(tmp_path):
    path = Path(tmp_path) / "documents.csv"
    common.write_catalog([FED_REF, ECB_REF], path)
    assert common.read_catalog(path) == [ECB_REF, FED_REF]  # sorted by bank, then date
