"""Unit tests for config, corpus loading, and the extraction runner (network-free)."""

import json
from datetime import date

import pytest

from rategauge import corpus
from rategauge.config import load_models
from rategauge.extract import runner
from rategauge.extract.clients import RawExtraction
from rategauge.sources import common
from rategauge.sources.common import DocumentRef

ECB_PAGE = """<html><body><main >
<div class="section"><p class="ecb-publicationDate">11 June 2026 </p>
<p>The Governing Council decided to raise the three key ECB interest rates
by 25 basis points.</p></div>
</main></body></html>"""

VALID_RECORD = {
    "bank": "ECB",
    "decision_date": "2026-06-11",
    "effective_date": "2026-06-17",
    "action": "hike",
    "change_bps": 25,
    "target_range_lower_pct": None,
    "target_range_upper_pct": None,
    "dfr_pct": 2.25,
    "mro_pct": 2.4,
    "mlf_pct": 2.65,
    "evidence_quote": "raise the three key ECB interest rates",
}


class TestConfig:
    def test_registry_loads_expected_models(self):
        models = load_models()
        assert {"gpt-5.4-nano", "gpt-5.4-mini", "claude-haiku-4-5"} <= set(models)
        assert models["claude-haiku-4-5"].provider == "anthropic"
        assert models["claude-haiku-4-5"].model_id == "claude-haiku-4-5-20251001"

    def test_cost_math_sync_and_batch(self):
        model = load_models()["gpt-5.4-nano"]  # $0.20 in / $1.25 out per MTok
        sync = model.cost_usd(1_000_000, 1_000_000)
        assert sync == pytest.approx(1.45)
        assert model.cost_usd(1_000_000, 1_000_000, batch=True) == pytest.approx(sync / 2)


class TestPrompts:
    def test_v001_loads(self):
        text = runner.load_prompt("v001")
        assert "no_policy_decision" in text
        assert "evidence_quote" in text

    def test_unknown_version_lists_available(self):
        with pytest.raises(FileNotFoundError, match="v001"):
            runner.load_prompt("v999")


class TestCorpus:
    def make_corpus(self, tmp_path):
        ref = DocumentRef("ECB", "ecb_mp260611", date(2026, 6, 11), "https://x/e", "decision")
        catalog = tmp_path / "documents.csv"
        common.write_catalog([ref], catalog)
        cache = common.DocumentCache(tmp_path / "cache")
        cache.put(ref, ECB_PAGE)
        return catalog, tmp_path / "cache"

    def test_load_documents_extracts_text(self, tmp_path):
        catalog, cache_dir = self.make_corpus(tmp_path)
        [doc] = corpus.load_documents(
            ["ecb_mp260611"], catalog_path=catalog, cache_dir=cache_dir
        )
        assert "raise the three key ECB interest rates" in doc.text
        assert doc.text.startswith("Published: 11 June 2026")  # date grounded
        assert doc.as_model_input().startswith("Source: European Central Bank")

    def test_missing_doc_fails_loudly(self, tmp_path):
        catalog, cache_dir = self.make_corpus(tmp_path)
        with pytest.raises(KeyError, match="rategauge ingest"):
            corpus.load_documents(["fed_nope"], catalog_path=catalog, cache_dir=cache_dir)

    def test_trap_catalog_merged_into_lookup(self, tmp_path):
        catalog, cache_dir = self.make_corpus(tmp_path)
        trap_ref = DocumentRef("ECB", "ecb_trap1", date(2025, 2, 6), "https://x/t", "non_decision")
        traps = tmp_path / "traps.csv"
        common.write_catalog([trap_ref], traps)
        common.DocumentCache(cache_dir).put(trap_ref, ECB_PAGE)
        [doc] = corpus.load_documents(
            ["ecb_trap1"], catalog_path=catalog, traps_path=traps, cache_dir=cache_dir
        )
        assert doc.ref.doc_type == "non_decision"
        # ECB non-decision docs are still press releases; the label stays honest.
        assert doc.as_model_input().startswith("Source: European Central Bank press release")

    def test_minutes_get_publication_source_label(self):
        ref = DocumentRef("FED", "fed_x", date(2024, 1, 1), "https://x", "minutes")
        doc = corpus.LoadedDocument(ref=ref, text="text")
        assert doc.as_model_input().startswith("Source: Federal Reserve (FOMC) publication")


class TestRunner:
    def run(self, tmp_path, monkeypatch, payloads, doc_ids=None):
        """Run the runner with a fake provider returning canned payloads.

        A payload that is an Exception instance is raised instead of returned.
        """
        doc_ids = doc_ids or tuple(f"doc{i}" for i in range(len(payloads)))

        def fake_load_documents(ids, **kwargs):
            return [
                corpus.LoadedDocument(
                    ref=DocumentRef("ECB", doc_id, date(2026, 6, 11), "https://x/e", "decision"),
                    text="doc text",
                )
                for doc_id in ids
            ]

        calls = iter(payloads)

        def fake_extract(client, model_id, prompt, document):
            payload = next(calls)
            if isinstance(payload, Exception):
                raise payload
            return RawExtraction(
                payload=payload, input_tokens=1000, output_tokens=200, latency_ms=5
            )

        monkeypatch.setattr(runner.corpus, "load_documents", fake_load_documents)
        monkeypatch.setattr(runner, "EXTRACTORS", {"openai": fake_extract})
        return runner.run_extraction(
            "gpt-5.4-nano",
            "v001",
            doc_ids,
            out_dir=tmp_path / "runs",
            ledger_path=tmp_path / "ledger.csv",
            client=object(),
        )

    def artifact_rows(self, tmp_path):
        artifact = tmp_path / "runs" / "gpt-5.4-nano__v001__s1.jsonl"
        return [
            json.loads(line)
            for line in artifact.read_text(encoding="utf-8").splitlines()
        ]

    def test_valid_payload_produces_ok_row_and_artifact(self, tmp_path, monkeypatch):
        rows = self.run(tmp_path, monkeypatch, [json.dumps(VALID_RECORD)])
        assert rows[0]["ok"] is True
        assert rows[0]["record"]["action"] == "hike"
        assert rows[0]["cost_usd"] > 0
        assert [row["doc_id"] for row in self.artifact_rows(tmp_path)] == ["doc0"]

    def test_schema_violation_recorded_not_raised(self, tmp_path, monkeypatch):
        bad = json.dumps({**VALID_RECORD, "action": "pivot"})
        rows = self.run(tmp_path, monkeypatch, [bad, "not json at all"])
        assert [row["ok"] for row in rows] == [False, False]
        assert "schema_violation" in rows[0]["error"]
        assert "schema_violation" in rows[1]["error"]
        # Cost is still recorded for failed extractions — tokens were spent.
        assert all(row["cost_usd"] > 0 for row in rows)

    def test_provider_error_recorded_and_run_continues(self, tmp_path, monkeypatch):
        from rategauge.extract.clients import EmptyResponseError

        payloads = [EmptyResponseError("no text block"), json.dumps(VALID_RECORD)]
        rows = self.run(tmp_path, monkeypatch, payloads)
        assert rows[0]["ok"] is False
        assert "api_error" in rows[0]["error"]
        assert rows[0]["cost_usd"] == 0.0
        assert rows[1]["ok"] is True  # the run survived the failure
        assert len(self.artifact_rows(tmp_path)) == 2

    def test_artifact_merges_by_doc_id_across_runs(self, tmp_path, monkeypatch):
        self.run(tmp_path, monkeypatch, [json.dumps(VALID_RECORD)] * 2, ("a", "b"))
        self.run(tmp_path, monkeypatch, [json.dumps(VALID_RECORD)], ("b",))
        rows = self.artifact_rows(tmp_path)
        assert sorted(row["doc_id"] for row in rows) == ["a", "b"]  # nothing truncated

    def test_ledger_records_token_and_cost_totals(self, tmp_path, monkeypatch):
        import csv

        self.run(tmp_path, monkeypatch, [json.dumps(VALID_RECORD)] * 3)
        with (tmp_path / "ledger.csv").open(newline="", encoding="utf-8") as handle:
            [entry] = list(csv.DictReader(handle))
        assert entry["documents"] == "3"
        assert entry["input_tokens"] == "3000"
        assert entry["output_tokens"] == "600"
        assert float(entry["cost_usd"]) > 0

    def test_empty_doc_ids_rejected(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="no doc_ids"):
            self.run(tmp_path, monkeypatch, [], doc_ids=())
