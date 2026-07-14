"""Unit tests for the FastAPI service (network-free via fake extractors)."""

import json

import pytest
from fastapi.testclient import TestClient

from rategauge import corpus
from rategauge.extract.clients import EmptyResponseError, RawExtraction
from rategauge.serve import api

VALID_RECORD = {
    "bank": "FED",
    "decision_date": "2024-09-18",
    "effective_date": None,
    "action": "cut",
    "change_bps": -50,
    "target_range_lower_pct": 4.75,
    "target_range_upper_pct": 5.0,
    "dfr_pct": None,
    "mro_pct": None,
    "mlf_pct": None,
    "evidence_quote": "lower the target range by 1/2 percentage point",
}

GRADED_ROWS = [
    {
        "doc_id": "fed_1", "bank": "FED", "model_key": "gpt-5.4-nano",
        "prompt_version": "v001", "announcement_date": "2024-09-18",
        "status": "graded", "expected_kind": "change", "action_correct": True,
        "fabricated_decision": None, "wrong_direction": False, "change_bps": "correct",
        "level": "correct", "effective_date": "abstained", "decision_date": "correct",
    },
    {
        "doc_id": "ecb_1", "bank": "ECB", "model_key": "gpt-5.4-nano",
        "prompt_version": "v001", "announcement_date": "2025-02-06",
        "status": "graded", "expected_kind": "hold", "action_correct": True,
        "fabricated_decision": False, "wrong_direction": None, "change_bps": None,
        "level": "correct", "effective_date": None, "decision_date": "abstained",
    },
    {
        "doc_id": "fed_2", "bank": "FED", "model_key": "gpt-5.4-nano",
        "prompt_version": "v001", "announcement_date": "2020-03-15",
        "status": "extraction_failed", "expected_kind": None, "action_correct": None,
        "fabricated_decision": None, "wrong_direction": None, "change_bps": None,
        "level": None, "effective_date": None, "decision_date": None,
    },
]


@pytest.fixture()
def client(tmp_path):
    graded_dir = tmp_path / "graded"
    runs_dir = tmp_path / "runs"
    scorecards_dir = tmp_path / "scorecards"
    for directory in (graded_dir, runs_dir, scorecards_dir):
        directory.mkdir()
    stem = "gpt-5.4-nano__v001__s1"
    with (graded_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as handle:
        for row in GRADED_ROWS:
            handle.write(json.dumps(row) + "\n")
    with (runs_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"doc_id": "fed_1", "record": VALID_RECORD}) + "\n")
    (scorecards_dir / f"{stem}.json").write_text(
        json.dumps({"model_key": "gpt-5.4-nano", "prompt_version": "v001", "graded": 3}),
        encoding="utf-8",
    )
    (scorecards_dir / "claude-haiku-4-5__v001__s1.json").write_text(
        json.dumps({"model_key": "claude-haiku-4-5", "prompt_version": "v001", "graded": 3}),
        encoding="utf-8",
    )
    app = api.create_app(
        graded_dir=graded_dir,
        scorecards_dir=scorecards_dir,
        runs_dir=runs_dir,
        traces_path=tmp_path / "traces.sqlite",
    )
    return TestClient(app)


def fake_extraction_env(monkeypatch, payloads):
    """Route provider calls to a fake extractor returning canned payloads."""
    calls = iter(payloads)
    seen_inputs = []

    def fake_extract(client, model_id, prompt, document):
        seen_inputs.append(document)
        payload = next(calls)
        if isinstance(payload, Exception):
            raise payload
        return RawExtraction(payload=payload, input_tokens=1000, output_tokens=200, latency_ms=7)

    monkeypatch.setattr(api, "EXTRACTORS", {"openai": fake_extract, "anthropic": fake_extract})
    monkeypatch.setattr(api, "CLIENT_BUILDERS", dict.fromkeys(("openai", "anthropic"), object))
    return seen_inputs


class TestReadEndpoints:
    def test_health(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["schema_version"] == "s1"

    def test_decisions_joins_record_and_filters(self, client):
        body = client.get("/decisions", params={"model": "gpt-5.4-nano", "bank": "FED"}).json()
        assert body["count"] == 2
        by_id = {row["doc_id"]: row for row in body["rows"]}
        assert by_id["fed_1"]["record"]["action"] == "cut"  # joined from the runs artifact
        assert by_id["fed_2"]["record"] is None

    def test_decisions_status_and_date_filters(self, client):
        body = client.get(
            "/decisions",
            params={"model": "gpt-5.4-nano", "status": "graded", "date_from": "2024-01-01"},
        ).json()
        assert [row["doc_id"] for row in body["rows"]] == ["fed_1", "ecb_1"]

    def test_decisions_unknown_model_404(self, client):
        assert client.get("/decisions", params={"model": "nope"}).status_code == 404

    def test_decisions_invalid_bank_422(self, client):
        response = client.get("/decisions", params={"model": "gpt-5.4-nano", "bank": "BOE"})
        assert response.status_code == 422

    def test_scorecards_and_model_filter(self, client):
        assert client.get("/eval/scorecard").json()["count"] == 2
        body = client.get("/eval/scorecard", params={"model": "gpt-5.4-nano"}).json()
        assert body["count"] == 1
        assert body["scorecards"][0]["model_key"] == "gpt-5.4-nano"
        assert client.get("/eval/scorecard", params={"model": "nope"}).status_code == 404


class TestExtract:
    def test_adhoc_text_extraction_traced(self, client, monkeypatch):
        inputs = fake_extraction_env(monkeypatch, [json.dumps(VALID_RECORD)])
        response = client.post(
            "/extract",
            json={"model": "gpt-5.4-nano", "bank": "FED", "text": "The Committee decided..."},
        )
        body = response.json()
        assert response.status_code == 200
        assert body["ok"] is True
        assert body["record"]["action"] == "cut"
        assert body["usage"]["cost_usd"] == pytest.approx(0.00045)  # nano, full price
        assert inputs[0].startswith("Source: Federal Reserve (FOMC) press release")

        [trace] = client.get("/traces").json()["traces"]
        assert trace["id"] == body["trace_id"]
        assert trace["ok"] == 1
        assert trace["doc_id"] == "adhoc"
        assert trace["cost_usd"] == pytest.approx(0.00045)

    def test_catalogued_doc_id_extraction(self, client, monkeypatch):
        from datetime import date

        from rategauge.sources.common import DocumentRef

        fake_extraction_env(monkeypatch, [json.dumps(VALID_RECORD)])
        ref = DocumentRef("ECB", "ecb_x", date(2025, 2, 6), "https://x", "decision")
        monkeypatch.setattr(
            api.corpus,
            "load_documents",
            lambda ids, **kwargs: [corpus.LoadedDocument(ref=ref, text="doc text")],
        )
        body = client.post("/extract", json={"model": "gpt-5.4-nano", "doc_id": "ecb_x"}).json()
        assert body["ok"] is True
        [trace] = client.get("/traces").json()["traces"]
        assert trace["doc_id"] == "ecb_x"
        assert trace["bank"] == "ECB"

    def test_provider_error_returns_ok_false_and_traces(self, client, monkeypatch):
        fake_extraction_env(monkeypatch, [EmptyResponseError("no text block")])
        body = client.post(
            "/extract", json={"model": "gpt-5.4-nano", "bank": "FED", "text": "x"}
        ).json()
        assert body["ok"] is False
        assert "api_error" in body["error"]
        assert body["usage"]["cost_usd"] == 0.0
        [trace] = client.get("/traces").json()["traces"]
        assert trace["ok"] == 0
        assert "api_error" in trace["error"]

    def test_schema_violation_still_costs(self, client, monkeypatch):
        fake_extraction_env(monkeypatch, ["not json at all"])
        body = client.post(
            "/extract", json={"model": "gpt-5.4-nano", "bank": "FED", "text": "x"}
        ).json()
        assert body["ok"] is False
        assert "schema_violation" in body["error"]
        assert body["usage"]["cost_usd"] > 0  # tokens were spent

    def test_unknown_model_422(self, client):
        response = client.post("/extract", json={"model": "gpt-9", "bank": "FED", "text": "x"})
        assert response.status_code == 422

    def test_path_traversal_rejected(self, client):
        # model/prompt_version become filename components; separators must 422.
        assert client.get("/decisions", params={"model": "../../secrets"}).status_code == 422
        assert (
            client.get(
                "/decisions", params={"model": "gpt-5.4-nano", "prompt_version": "../x"}
            ).status_code
            == 422
        )
        response = client.post(
            "/extract",
            json={"model": "gpt-5.4-nano", "bank": "FED", "text": "x",
                  "prompt_version": "../../../README"},
        )
        assert response.status_code == 422

    def test_billed_empty_response_traces_real_spend(self, client, monkeypatch):
        # An empty Anthropic response is still billed; the trace must not say $0.
        fake_extraction_env(
            monkeypatch,
            [EmptyResponseError("no text block", input_tokens=1000, output_tokens=50)],
        )
        body = client.post(
            "/extract", json={"model": "gpt-5.4-nano", "bank": "FED", "text": "x"}
        ).json()
        assert body["ok"] is False
        assert body["usage"]["input_tokens"] == 1000
        assert body["usage"]["cost_usd"] > 0
        [trace] = client.get("/traces").json()["traces"]
        assert trace["cost_usd"] > 0

    def test_unwritable_trace_db_does_not_lose_the_paid_result(self, client, monkeypatch):
        import sqlite3

        fake_extraction_env(monkeypatch, [json.dumps(VALID_RECORD)])

        def broken_record_trace(path, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(api.tracing, "record_trace", broken_record_trace)
        body = client.post(
            "/extract", json={"model": "gpt-5.4-nano", "bank": "FED", "text": "x"}
        ).json()
        assert body["ok"] is True  # the paid record still reaches the caller
        assert body["record"]["action"] == "cut"
        assert body["trace_id"] is None

    def test_unexpected_extractor_exception_still_traced(self, client, monkeypatch):
        fake_extraction_env(monkeypatch, [RuntimeError("usage was None")])
        with pytest.raises(RuntimeError, match="usage was None"):
            client.post("/extract", json={"model": "gpt-5.4-nano", "bank": "FED", "text": "x"})
        [trace] = client.get("/traces").json()["traces"]
        assert trace["ok"] == 0
        assert "unexpected" in trace["error"]

    def test_unknown_doc_id_404(self, client, monkeypatch):
        fake_extraction_env(monkeypatch, [])
        response = client.post("/extract", json={"model": "gpt-5.4-nano", "doc_id": "nope"})
        assert response.status_code == 404

    def test_text_without_bank_422(self, client, monkeypatch):
        fake_extraction_env(monkeypatch, [])
        response = client.post("/extract", json={"model": "gpt-5.4-nano", "text": "x"})
        assert response.status_code == 422
