"""Batch extraction: submit / status / collect via the providers' Batch APIs.

Both providers price batch tokens at 50% of the synchronous rate, so the
full-corpus league-table runs go through here (the Anthropic budget requires
it). Batches finish asynchronously (minutes to 24h), so the lifecycle is
split into idempotent CLI steps: ``submit`` uploads provider requests and
records a state file under eval/batches/; ``status`` refreshes provider-side
progress; ``collect`` parses finished results into the same JSONL artifacts
and cost ledger the synchronous runner writes, so grading is identical for
both paths.
"""

import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from rategauge import corpus
from rategauge.config import ModelConfig, load_credentials, load_models
from rategauge.extract.clients import CLIENT_BUILDERS, MAX_OUTPUT_TOKENS
from rategauge.extract.runner import (
    LEDGER_PATH,
    RUNS_DIR,
    append_ledger,
    apply_payload,
    load_prompt,
    new_row,
    write_artifact,
)
from rategauge.schema import EXTRACTION_JSON_SCHEMA, SCHEMA_VERSION

logger = logging.getLogger(__name__)

BATCHES_DIR = Path("eval/batches")

# Provider-side lifecycle states after which no further progress will happen.
TERMINAL_STATUSES = {
    "openai": {"completed", "failed", "expired", "cancelled"},
    "anthropic": {"ended"},
}

# OpenAI error-file codes for requests that never executed (and were never
# billed) — routed to missing_doc_ids for resubmission, like Anthropic's
# canceled/expired result types, instead of clobbering artifacts with error rows.
UNBILLED_OPENAI_ERROR_CODES = {"batch_expired", "batch_cancelled"}


class BatchNotReady(Exception):
    """The provider is still processing the batch; collect again later."""


def submit_batch(
    model_key: str,
    prompt_version: str,
    doc_ids: tuple[str, ...],
    *,
    state_dir: Path = BATCHES_DIR,
    client: object | None = None,
    force: bool = False,
) -> dict:
    """Submit one (model, prompt) batch over the given documents; persist state.

    The state file carries everything ``collect`` needs later (doc metadata for
    artifact rows, provider ids), so collection works from a fresh process.
    Refuses to double-spend on a (model, prompt) that already has an
    uncollected batch unless ``force`` is set.
    """
    if not doc_ids:
        raise ValueError("no doc_ids given")
    pending = [
        path.stem
        for path in list_states(state_dir)
        if (existing := load_state(path))["model_key"] == model_key
        and existing["prompt_version"] == prompt_version
        and not existing["collected"]
    ]
    if pending and not force:
        raise ValueError(
            f"uncollected batch(es) for {model_key}/{prompt_version} already exist: "
            f"{pending} — collect them first, or pass --force to submit anyway"
        )
    load_credentials()
    model = load_models()[model_key]
    prompt = load_prompt(prompt_version)
    documents = corpus.load_documents(doc_ids)
    client = client or CLIENT_BUILDERS[model.provider]()

    # chars/4 is a crude token proxy — good enough for a budget sanity line,
    # logged BEFORE money is committed at the provider.
    estimated_input_tokens = (
        sum(len(prompt) + len(doc.as_model_input()) for doc in documents) // 4
    )
    logger.info(
        "submitting %s/%s: %d documents, ~%d estimated input tokens",
        model_key,
        prompt_version,
        len(documents),
        estimated_input_tokens,
    )
    batch_id, provider_ids = _SUBMITTERS[model.provider](client, model, prompt, documents)
    # Log the id before touching the filesystem: if the state write fails, the
    # paid-for batch must still be findable without the provider dashboard.
    logger.info("provider accepted batch %s — persisting state", batch_id)

    state = {
        "batch_id": batch_id,
        "provider": model.provider,
        "model_key": model_key,
        "prompt_version": prompt_version,
        "schema_version": SCHEMA_VERSION,
        "submitted_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "submitted",
        "collected": False,
        "estimated_input_tokens": estimated_input_tokens,
        "documents": [
            {
                "doc_id": doc.ref.doc_id,
                "bank": doc.ref.bank,
                "announcement_date": doc.ref.announcement_date.isoformat(),
            }
            for doc in documents
        ],
        **provider_ids,
    }
    _save_state(state, state_dir / f"{batch_id}.json")
    return state


def _submit_openai(client, model: ModelConfig, prompt: str, documents) -> tuple[str, dict]:
    lines = [
        json.dumps(
            {
                "custom_id": document.ref.doc_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model.model_id,
                    "instructions": prompt,
                    "input": document.as_model_input(),
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "rate_decision",
                            "schema": EXTRACTION_JSON_SCHEMA,
                            "strict": True,
                        }
                    },
                },
            }
        )
        for document in documents
    ]
    upload = client.files.create(
        file=("rategauge_batch.jsonl", "\n".join(lines).encode("utf-8")),
        purpose="batch",
    )
    batch = client.batches.create(
        input_file_id=upload.id, endpoint="/v1/responses", completion_window="24h"
    )
    return batch.id, {"openai_input_file_id": upload.id}


def _submit_anthropic(client, model: ModelConfig, prompt: str, documents) -> tuple[str, dict]:
    requests = [
        {
            "custom_id": document.ref.doc_id,
            "params": {
                "model": model.model_id,
                "max_tokens": MAX_OUTPUT_TOKENS,
                "system": prompt,
                "output_config": {
                    "format": {"type": "json_schema", "schema": EXTRACTION_JSON_SCHEMA}
                },
                "messages": [{"role": "user", "content": document.as_model_input()}],
            },
        }
        for document in documents
    ]
    batch = client.messages.batches.create(requests=requests)
    return batch.id, {}


_SUBMITTERS = {"openai": _submit_openai, "anthropic": _submit_anthropic}


def refresh_status(state_path: Path, *, client: object | None = None) -> dict:
    """Query the provider for current batch progress and persist it."""
    state = load_state(state_path)
    if state["collected"]:
        return state
    load_credentials()
    client = client or CLIENT_BUILDERS[state["provider"]]()
    if state["provider"] == "openai":
        batch = client.batches.retrieve(state["batch_id"])
        state["status"] = batch.status
        state["openai_output_file_id"] = batch.output_file_id
        state["openai_error_file_id"] = batch.error_file_id
        if batch.status == "failed" and getattr(batch, "errors", None):
            state["openai_errors"] = str(batch.errors)[:500]
    else:
        batch = client.messages.batches.retrieve(state["batch_id"])
        state["status"] = batch.processing_status
    _save_state(state, state_path)
    return state


def is_terminal(state: dict) -> bool:
    return state["status"] in TERMINAL_STATUSES[state["provider"]]


def collect_batch(
    state_path: Path,
    *,
    out_dir: Path = RUNS_DIR,
    ledger_path: Path = LEDGER_PATH,
    client: object | None = None,
) -> list[dict]:
    """Parse a finished batch into artifact rows; merge artifact + cost ledger.

    Documents with no billed result (expired/canceled requests) are reported in
    the state file's ``missing_doc_ids`` instead of being written as error rows:
    an error row would clobber a good earlier result on artifact merge, and the
    right remedy for an unbilled request is resubmission.
    """
    state = load_state(state_path)
    if state["collected"]:
        raise ValueError(f"{state['batch_id']} already collected")
    if state["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError(
            f"{state['batch_id']} was submitted under schema {state['schema_version']!r} "
            f"but the current schema is {SCHEMA_VERSION!r} — refusing to mix generations "
            f"in one artifact"
        )
    load_credentials()
    client = client or CLIENT_BUILDERS[state["provider"]]()
    state = refresh_status(state_path, client=client)
    if not is_terminal(state):
        raise BatchNotReady(f"{state['batch_id']} is {state['status']}")

    model = load_models()[state["model_key"]]
    docs = {doc["doc_id"]: doc for doc in state["documents"]}
    if state["provider"] == "openai":
        rows = _collect_openai(client, state, model, docs)
    else:
        rows = _collect_anthropic(client, state, model, docs)

    missing = sorted(set(docs) - {row["doc_id"] for row in rows})
    if missing:
        logger.warning(
            "%d documents missing from %s output (resubmit them): %s",
            len(missing),
            state["batch_id"],
            missing,
        )
    if rows:
        write_artifact(rows, state["model_key"], state["prompt_version"], out_dir)
        if _ledger_has_run(ledger_path, state["batch_id"]):
            # Collect retry after a crash between ledger append and state save —
            # never double-count spend against the budget.
            logger.info("%s already in the cost ledger — skipping duplicate entry",
                        state["batch_id"])
        else:
            append_ledger(
                rows, model, state["prompt_version"], ledger_path, run_id=state["batch_id"]
            )
    state["collected"] = True
    state["collected_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    state["missing_doc_ids"] = missing
    _save_state(state, state_path)
    return rows


def _collect_openai(client, state: dict, model: ModelConfig, docs: dict) -> list[dict]:
    output_file_id = state.get("openai_output_file_id")
    error_file_id = state.get("openai_error_file_id")
    if not output_file_id and not error_file_id:
        # e.g. status "failed" before any request executed (validation error,
        # quota). Nothing was billed: return no rows so every document lands in
        # missing_doc_ids and the state is retired instead of wedging collect.
        logger.error(
            "%s finished as %s with no output or error file (errors: %s)",
            state["batch_id"],
            state["status"],
            state.get("openai_errors"),
        )
        return []
    rows: dict[str, dict] = {}
    if output_file_id:
        for result in _read_jsonl(client.files.content(output_file_id).text):
            row = _base_row(docs[result["custom_id"]], state)
            response = result.get("response")
            if result.get("error") or not response or response.get("status_code") != 200:
                row["error"] = f"api_error: {json.dumps(_openai_error_detail(result))[:300]}"
            else:
                body = response["body"]
                usage = body.get("usage") or {}
                row["input_tokens"] = usage.get("input_tokens", 0)
                row["output_tokens"] = usage.get("output_tokens", 0)
                row["cost_usd"] = round(
                    model.cost_usd(row["input_tokens"], row["output_tokens"], batch=True), 6
                )
                payload = _openai_output_text(body)
                if payload is None:
                    row["error"] = "api_error: no output_text in batch response"
                else:
                    apply_payload(row, payload)
            rows[row["doc_id"]] = row
    # Requests that failed (or never executed) land in a separate error file.
    if error_file_id:
        for result in _read_jsonl(client.files.content(error_file_id).text):
            doc_id = result["custom_id"]
            if doc_id in rows:
                continue
            detail = _openai_error_detail(result)
            if detail.get("code") in UNBILLED_OPENAI_ERROR_CODES:
                continue  # never executed, never billed -> missing_doc_ids
            row = _base_row(docs[doc_id], state)
            row["error"] = f"api_error: {json.dumps(detail)[:300]}"
            rows[doc_id] = row
    return list(rows.values())


def _openai_error_detail(result: dict) -> dict:
    """The most specific error payload in a batch result line.

    The top-level ``error`` is populated for non-HTTP failures (e.g.
    batch_expired); for HTTP failures it is null and the diagnostic lives in
    ``response.body.error``.
    """
    response = result.get("response") or {}
    body = response.get("body") or {}
    return (
        result.get("error")
        or body.get("error")
        or {"status_code": response.get("status_code")}
    )


def _collect_anthropic(client, state: dict, model: ModelConfig, docs: dict) -> list[dict]:
    rows: list[dict] = []
    for result in client.messages.batches.results(state["batch_id"]):
        outcome = result.result
        if outcome.type in ("canceled", "expired"):
            continue  # not billed; surfaces in missing_doc_ids for resubmission
        row = _base_row(docs[result.custom_id], state)
        if outcome.type != "succeeded":
            detail = getattr(outcome, "error", None)
            row["error"] = f"api_error: batch result {outcome.type}: {detail}"
            rows.append(row)
            continue
        message = outcome.message
        row["input_tokens"] = message.usage.input_tokens
        row["output_tokens"] = message.usage.output_tokens
        row["cost_usd"] = round(
            model.cost_usd(row["input_tokens"], row["output_tokens"], batch=True), 6
        )
        payload = next((block.text for block in message.content if block.type == "text"), None)
        if payload is None:
            row["error"] = f"api_error: no text block (stop_reason={message.stop_reason})"
        else:
            apply_payload(row, payload)
        rows.append(row)
    return rows


def _base_row(doc: dict, state: dict) -> dict:
    # collect_batch guarantees state["schema_version"] == current SCHEMA_VERSION,
    # so new_row's stamp is already correct.
    return new_row(
        doc["doc_id"],
        doc["bank"],
        doc["announcement_date"],
        state["model_key"],
        state["prompt_version"],
    )


def _openai_output_text(body: dict) -> str | None:
    parts = [
        block["text"]
        for item in body.get("output") or []
        if item.get("type") == "message"
        for block in item.get("content") or []
        if block.get("type") == "output_text"
    ]
    return "".join(parts) or None


def _read_jsonl(text: str):
    for line in text.splitlines():
        if line.strip():
            yield json.loads(line)


def load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)  # atomic swap: a crash mid-write can't corrupt the state


def _ledger_has_run(ledger_path: Path, run_id: str) -> bool:
    if not ledger_path.exists():
        return False
    with ledger_path.open(newline="", encoding="utf-8") as handle:
        return any(entry.get("run_id") == run_id for entry in csv.DictReader(handle))


def list_states(state_dir: Path = BATCHES_DIR) -> list[Path]:
    if not state_dir.exists():
        return []
    return sorted(state_dir.glob("*.json"))
