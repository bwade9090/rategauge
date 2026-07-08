"""Unit tests for batch submit/status/collect (network-free via fake provider clients)."""

import json
from datetime import date
from types import SimpleNamespace

import pytest

from rategauge import corpus
from rategauge.extract import batch
from rategauge.sources.common import DocumentRef

# Mirrors tests/test_extract.py — a record that passes RateDecision validation.
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

# gpt-5.4-nano at batch (50%) rate: (1000 * $0.20 + 200 * $1.25) / 1e6 / 2
NANO_BATCH_COST = pytest.approx(0.000225)


def fake_load_documents(doc_ids, **kwargs):
    return [
        corpus.LoadedDocument(
            ref=DocumentRef("ECB", doc_id, date(2026, 6, 11), f"https://x/{doc_id}", "decision"),
            text="doc text",
        )
        for doc_id in doc_ids
    ]


class FakeOpenAI:
    """Just enough of the OpenAI SDK surface for batch submit + collect."""

    def __init__(self, status="completed", output_lines=None, error_lines=None):
        self.uploaded = None
        self.created = None
        self._status = status
        self._files = {}
        if output_lines is not None:
            self._files["file_out"] = "\n".join(output_lines)
        if error_lines is not None:
            self._files["file_err"] = "\n".join(error_lines)
        self.files = SimpleNamespace(create=self._file_create, content=self._file_content)
        self.batches = SimpleNamespace(create=self._batch_create, retrieve=self._batch_retrieve)

    def _file_create(self, file, purpose):
        assert purpose == "batch"
        self.uploaded = file
        return SimpleNamespace(id="file_in")

    def _file_content(self, file_id):
        return SimpleNamespace(text=self._files[file_id])

    def _batch_create(self, input_file_id, endpoint, completion_window):
        self.created = {"input_file_id": input_file_id, "endpoint": endpoint}
        return SimpleNamespace(id="batch_x1", status="validating")

    def _batch_retrieve(self, batch_id):
        return SimpleNamespace(
            id=batch_id,
            status=self._status,
            output_file_id="file_out" if "file_out" in self._files else None,
            error_file_id="file_err" if "file_err" in self._files else None,
        )


def openai_ok_line(doc_id, payload):
    return json.dumps(
        {
            "custom_id": doc_id,
            "response": {
                "status_code": 200,
                "body": {
                    "output": [
                        {"type": "reasoning", "content": []},
                        {"type": "message", "content": [{"type": "output_text", "text": payload}]},
                    ],
                    "usage": {"input_tokens": 1000, "output_tokens": 200},
                },
            },
            "error": None,
        }
    )


class FakeAnthropic:
    """Just enough of the Anthropic SDK surface for batch submit + collect."""

    def __init__(self, processing_status="ended", results=()):
        self.requests = None
        self._status = processing_status
        self._resultset = results
        self.messages = SimpleNamespace(
            batches=SimpleNamespace(
                create=self._create, retrieve=self._retrieve, results=self._results
            )
        )

    def _create(self, requests):
        self.requests = requests
        return SimpleNamespace(id="msgbatch_x1", processing_status="in_progress")

    def _retrieve(self, batch_id):
        return SimpleNamespace(id=batch_id, processing_status=self._status)

    def _results(self, batch_id):
        return iter(self._resultset)


def anthropic_result(doc_id, payload=None, result_type="succeeded"):
    if result_type != "succeeded":
        return SimpleNamespace(
            custom_id=doc_id,
            result=SimpleNamespace(type=result_type, error={"type": "api_error"}),
        )
    return SimpleNamespace(
        custom_id=doc_id,
        result=SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(
                content=[SimpleNamespace(type="text", text=payload)],
                usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
                stop_reason="end_turn",
            ),
        ),
    )


@pytest.fixture(autouse=True)
def _fake_corpus(monkeypatch):
    monkeypatch.setattr(batch.corpus, "load_documents", fake_load_documents)


class TestSubmit:
    def test_openai_uploads_schema_constrained_requests_and_writes_state(self, tmp_path):
        client = FakeOpenAI()
        batch.submit_batch("gpt-5.4-nano", "v001", ("a", "b"), state_dir=tmp_path, client=client)

        _, content = self.uploaded_file(client)
        lines = [json.loads(line) for line in content.decode("utf-8").splitlines()]
        assert [line["custom_id"] for line in lines] == ["a", "b"]
        assert lines[0]["url"] == "/v1/responses"
        body = lines[0]["body"]
        assert body["model"] == "gpt-5.4-nano"
        assert body["text"]["format"]["strict"] is True
        assert body["text"]["format"]["schema"]["properties"]["action"]
        assert "doc text" in body["input"]
        assert body["instructions"]  # system prompt attached
        assert client.created == {"input_file_id": "file_in", "endpoint": "/v1/responses"}

        state = batch.load_state(tmp_path / "batch_x1.json")
        assert state["provider"] == "openai"
        assert state["openai_input_file_id"] == "file_in"
        assert state["collected"] is False
        assert state["estimated_input_tokens"] > 0
        assert state["documents"][0] == {
            "doc_id": "a",
            "bank": "ECB",
            "announcement_date": "2026-06-11",
        }

    def uploaded_file(self, client):
        assert client.uploaded is not None
        return client.uploaded

    def test_anthropic_builds_schema_constrained_requests(self, tmp_path):
        client = FakeAnthropic()
        batch.submit_batch("claude-haiku-4-5", "v001", ("a",), state_dir=tmp_path, client=client)

        [request] = client.requests
        assert request["custom_id"] == "a"
        params = request["params"]
        assert params["model"] == "claude-haiku-4-5-20251001"
        assert params["output_config"]["format"]["type"] == "json_schema"
        assert params["system"]  # system prompt attached
        assert "doc text" in params["messages"][0]["content"]
        assert batch.load_state(tmp_path / "msgbatch_x1.json")["provider"] == "anthropic"

    def test_empty_doc_ids_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="no doc_ids"):
            batch.submit_batch("gpt-5.4-nano", "v001", (), state_dir=tmp_path)

    def test_pending_duplicate_rejected_unless_forced(self, tmp_path):
        batch.submit_batch("gpt-5.4-nano", "v001", ("a",), state_dir=tmp_path, client=FakeOpenAI())
        # Same (model, prompt) with an uncollected batch pending: refuse to double-spend.
        with pytest.raises(ValueError, match="collect them first"):
            batch.submit_batch(
                "gpt-5.4-nano", "v001", ("a",), state_dir=tmp_path, client=FakeOpenAI()
            )
        # A different model is unaffected, and --force overrides.
        batch.submit_batch(
            "claude-haiku-4-5", "v001", ("a",), state_dir=tmp_path, client=FakeAnthropic()
        )
        batch.submit_batch(
            "gpt-5.4-nano", "v001", ("a",), state_dir=tmp_path, client=FakeOpenAI(), force=True
        )


class TestCollect:
    def submit(self, tmp_path, client, model_key, doc_ids):
        state = batch.submit_batch(
            model_key, "v001", doc_ids, state_dir=tmp_path / "batches", client=client
        )
        return tmp_path / "batches" / f"{state['batch_id']}.json"

    def collect(self, tmp_path, path, client):
        return batch.collect_batch(
            path,
            out_dir=tmp_path / "runs",
            ledger_path=tmp_path / "ledger.csv",
            client=client,
        )

    def artifact_rows(self, tmp_path, stem):
        artifact = tmp_path / "runs" / f"{stem}.jsonl"
        return [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines()]

    def test_openai_collect_writes_artifact_ledger_and_marks_state(self, tmp_path):
        client = FakeOpenAI(
            status="completed",
            output_lines=[openai_ok_line("a", json.dumps(VALID_RECORD))],
            error_lines=[
                json.dumps({"custom_id": "b", "response": None, "error": {"message": "boom"}})
            ],
        )
        path = self.submit(tmp_path, client, "gpt-5.4-nano", ("a", "b", "c"))
        rows = self.collect(tmp_path, path, client)

        by_id = {row["doc_id"]: row for row in rows}
        assert by_id["a"]["ok"] is True
        assert by_id["a"]["record"]["action"] == "hike"
        assert by_id["a"]["cost_usd"] == NANO_BATCH_COST  # 50% batch discount applied
        assert "boom" in by_id["b"]["error"]
        assert "c" not in by_id  # no billed result -> reported missing, not an error row

        state = batch.load_state(path)
        assert state["collected"] is True
        assert state["missing_doc_ids"] == ["c"]
        assert len(self.artifact_rows(tmp_path, "gpt-5.4-nano__v001__s1")) == 2
        ledger = (tmp_path / "ledger.csv").read_text(encoding="utf-8")
        assert "cost_usd" in ledger
        assert "batch_x1" in ledger  # run_id recorded for idempotent retries

    def test_openai_schema_violation_recorded(self, tmp_path):
        client = FakeOpenAI(status="completed", output_lines=[openai_ok_line("a", "not json")])
        path = self.submit(tmp_path, client, "gpt-5.4-nano", ("a",))
        [row] = self.collect(tmp_path, path, client)
        assert row["ok"] is False
        assert "schema_violation" in row["error"]
        assert row["cost_usd"] == NANO_BATCH_COST  # tokens were still spent

    def test_anthropic_collect_handles_success_error_and_expiry(self, tmp_path):
        client = FakeAnthropic(
            results=(
                anthropic_result("a", json.dumps(VALID_RECORD)),
                anthropic_result("b", result_type="errored"),
                anthropic_result("c", result_type="expired"),
            )
        )
        path = self.submit(tmp_path, client, "claude-haiku-4-5", ("a", "b", "c"))
        rows = self.collect(tmp_path, path, client)

        by_id = {row["doc_id"]: row for row in rows}
        assert by_id["a"]["ok"] is True
        assert by_id["a"]["cost_usd"] == pytest.approx((1000 * 1.00 + 200 * 5.00) / 1e6 / 2)
        assert "errored" in by_id["b"]["error"]
        assert "c" not in by_id
        assert batch.load_state(path)["missing_doc_ids"] == ["c"]

    def test_not_ready_raises_and_leaves_state_collectable(self, tmp_path):
        client = FakeOpenAI(status="in_progress")
        path = self.submit(tmp_path, client, "gpt-5.4-nano", ("a",))
        with pytest.raises(batch.BatchNotReady, match="in_progress"):
            self.collect(tmp_path, path, client)
        assert batch.load_state(path)["collected"] is False

    def test_collect_twice_rejected(self, tmp_path):
        client = FakeOpenAI(
            status="completed", output_lines=[openai_ok_line("a", json.dumps(VALID_RECORD))]
        )
        path = self.submit(tmp_path, client, "gpt-5.4-nano", ("a",))
        self.collect(tmp_path, path, client)
        with pytest.raises(ValueError, match="already collected"):
            self.collect(tmp_path, path, client)

    def test_openai_failed_batch_retires_state_with_all_docs_missing(self, tmp_path):
        # status "failed" before any request executed: no output/error file,
        # nothing billed. Collect must retire the batch, not wedge forever.
        client = FakeOpenAI(status="failed")
        path = self.submit(tmp_path, client, "gpt-5.4-nano", ("a", "b"))
        rows = self.collect(tmp_path, path, client)
        assert rows == []
        state = batch.load_state(path)
        assert state["collected"] is True
        assert state["missing_doc_ids"] == ["a", "b"]
        assert not (tmp_path / "runs").exists()  # no artifact, no ledger entry

    def test_openai_expired_requests_route_to_missing_not_error_rows(self, tmp_path):
        # Unfinished requests of an expired batch are unbilled and land in the
        # error file with code batch_expired — they must NOT become error rows
        # (which would clobber paid-for rows on artifact merge).
        client = FakeOpenAI(
            status="expired",
            output_lines=[openai_ok_line("a", json.dumps(VALID_RECORD))],
            error_lines=[
                json.dumps(
                    {
                        "custom_id": "b",
                        "response": None,
                        "error": {"code": "batch_expired", "message": "not executed"},
                    }
                )
            ],
        )
        path = self.submit(tmp_path, client, "gpt-5.4-nano", ("a", "b"))
        rows = self.collect(tmp_path, path, client)
        assert [row["doc_id"] for row in rows] == ["a"]
        assert rows[0]["ok"] is True
        assert batch.load_state(path)["missing_doc_ids"] == ["b"]

    def test_openai_incomplete_response_classified_as_api_error(self, tmp_path):
        # A content-filtered response carries a truncated output_text stump;
        # it must be classified as a provider failure, not a schema violation.
        line = json.dumps(
            {
                "custom_id": "a",
                "response": {
                    "status_code": 200,
                    "body": {
                        "status": "incomplete",
                        "incomplete_details": {"reason": "content_filter"},
                        "output": [
                            {
                                "type": "message",
                                "content": [{"type": "output_text", "text": '{"bank": "FED'}],
                            }
                        ],
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
                "error": None,
            }
        )
        client = FakeOpenAI(status="completed", output_lines=[line])
        path = self.submit(tmp_path, client, "gpt-5.4-nano", ("a",))
        [row] = self.collect(tmp_path, path, client)
        assert row["ok"] is False
        assert row["error"] == "api_error: response incomplete (content_filter)"
        assert row["cost_usd"] == 0.0  # filtered responses are not billed

    def test_openai_http_error_detail_read_from_response_body(self, tmp_path):
        # For HTTP failures the top-level error is null; the reason lives in
        # response.body.error and must survive into the artifact row.
        client = FakeOpenAI(
            status="completed",
            error_lines=[
                json.dumps(
                    {
                        "custom_id": "a",
                        "response": {
                            "status_code": 400,
                            "body": {"error": {"code": "invalid_value", "message": "bad schema"}},
                        },
                        "error": None,
                    }
                )
            ],
        )
        path = self.submit(tmp_path, client, "gpt-5.4-nano", ("a",))
        [row] = self.collect(tmp_path, path, client)
        assert "bad schema" in row["error"]

    def test_ledger_not_duplicated_when_recollecting_after_crash(self, tmp_path):
        import csv

        client = FakeOpenAI(
            status="completed", output_lines=[openai_ok_line("a", json.dumps(VALID_RECORD))]
        )
        path = self.submit(tmp_path, client, "gpt-5.4-nano", ("a",))
        self.collect(tmp_path, path, client)
        # Simulate a crash after append_ledger but before the state save:
        # on disk the batch still looks uncollected, so the user retries.
        state = batch.load_state(path)
        state["collected"] = False
        path.write_text(json.dumps(state), encoding="utf-8")
        self.collect(tmp_path, path, client)

        with (tmp_path / "ledger.csv").open(newline="", encoding="utf-8") as handle:
            entries = list(csv.DictReader(handle))
        assert len(entries) == 1  # the retry must not double-count spend
        assert entries[0]["run_id"] == "batch_x1"

    def test_schema_version_mismatch_refused(self, tmp_path):
        client = FakeOpenAI(
            status="completed", output_lines=[openai_ok_line("a", json.dumps(VALID_RECORD))]
        )
        path = self.submit(tmp_path, client, "gpt-5.4-nano", ("a",))
        state = batch.load_state(path)
        state["schema_version"] = "s0"
        path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(RuntimeError, match="schema"):
            self.collect(tmp_path, path, client)
        assert batch.load_state(path)["collected"] is False


class TestStatus:
    def test_refresh_updates_status_and_file_ids(self, tmp_path):
        client = FakeOpenAI(status="completed", output_lines=[], error_lines=[])
        state = batch.submit_batch(
            "gpt-5.4-nano", "v001", ("a",), state_dir=tmp_path, client=client
        )
        path = tmp_path / f"{state['batch_id']}.json"
        refreshed = batch.refresh_status(path, client=client)
        assert refreshed["status"] == "completed"
        assert refreshed["openai_output_file_id"] == "file_out"
        assert refreshed["openai_error_file_id"] == "file_err"
        assert batch.load_state(path)["status"] == "completed"  # persisted

    def test_refresh_skips_provider_call_once_collected(self, tmp_path):
        client = FakeOpenAI(
            status="completed", output_lines=[openai_ok_line("a", json.dumps(VALID_RECORD))]
        )
        state = batch.submit_batch(
            "gpt-5.4-nano", "v001", ("a",), state_dir=tmp_path, client=client
        )
        path = tmp_path / f"{state['batch_id']}.json"
        batch.collect_batch(
            path, out_dir=tmp_path / "runs", ledger_path=tmp_path / "l.csv", client=client
        )
        exploding = object()  # any attribute access would raise AttributeError
        assert batch.refresh_status(path, client=exploding)["collected"] is True

    def test_list_states_empty_when_missing(self, tmp_path):
        assert batch.list_states(tmp_path / "nope") == []
