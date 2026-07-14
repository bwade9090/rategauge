"""Extraction runner: documents x model x prompt -> JSONL artifacts + cost ledger.

Every run is identified by (model_key, prompt_version, schema_version) and
writes one JSONL artifact under eval/runs/. Evaluation consumes artifacts, so
scoring is replayable offline. A cost ledger row is appended per run.
"""

import csv
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import anthropic
import openai
from pydantic import ValidationError

from rategauge import corpus
from rategauge.config import ModelConfig, load_credentials, load_models
from rategauge.extract.clients import CLIENT_BUILDERS, EXTRACTORS, EmptyResponseError
from rategauge.schema import SCHEMA_VERSION, RateDecision

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
RUNS_DIR = Path("eval/runs")
LEDGER_PATH = Path("eval/cost_ledger.csv")

# Representative development subset: every URL/template era of both banks,
# hikes, cuts, holds, and emergency decisions. Kept small for cheap iteration.
DEV_SET = (
    "fed_20260617a",  # modern hold
    "fed_20240918a",  # modern -50bp cut
    "fed_20200315a",  # emergency -100bp cut (Sunday announcement)
    "fed_20081216b",  # cut to a 0-0.25 target range (first range statement)
    "fed_20050202",  # boarddocs-era +25bp hike
    "fed_20021106",  # boarddocs-era -50bp cut
    "ecb_mp260611",  # modern +25bp hike
    "ecb_mp240912",  # modern cut with corridor adjustment (DFR -25, MRO -60)
    "ecb_pr121206",  # 2012-era hold
    "ecb_pr081008",  # 2008 coordinated -50bp cut
    "ecb_pr050804_2",  # minimum-bid-rate-era hold
    "ecb_pr991202",  # 1999-era hold with orderedlist layout
)


def load_prompt(version: str) -> str:
    path = PROMPTS_DIR / f"{version}.md"
    if not path.exists():
        available = sorted(p.stem for p in PROMPTS_DIR.glob("*.md"))
        raise FileNotFoundError(f"prompt {version!r} not found; available: {available}")
    return path.read_text(encoding="utf-8")


def new_row(
    doc_id: str, bank: str, announcement_date: str, model_key: str, prompt_version: str
) -> dict:
    """A blank artifact row; extraction outcome fields are filled in afterwards."""
    return {
        "doc_id": doc_id,
        "bank": bank,
        "announcement_date": announcement_date,
        "model_key": model_key,
        "prompt_version": prompt_version,
        "schema_version": SCHEMA_VERSION,
        "ok": False,
        "record": None,
        "error": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "latency_ms": 0,
    }


def apply_payload(row: dict, payload: str) -> None:
    """Validate a model payload into the row: ok+record, or a schema_violation error."""
    try:
        record = RateDecision.model_validate(json.loads(payload))
        row["ok"] = True
        row["record"] = json.loads(record.model_dump_json())
    except (json.JSONDecodeError, ValidationError) as error:
        row["error"] = f"schema_violation: {error}"
        logger.warning("schema violation for %s: %s", row["doc_id"], error)


def run_extraction(
    model_key: str,
    prompt_version: str,
    doc_ids: tuple[str, ...],
    *,
    out_dir: Path = RUNS_DIR,
    ledger_path: Path = LEDGER_PATH,
    client: object | None = None,
) -> list[dict]:
    """Run one extraction pass and persist the artifact. Returns the new rows.

    Artifacts are merged by doc_id (re-running a subset updates those rows and
    keeps the rest); rows and the cost ledger are persisted even if the run
    aborts mid-way, so paid-for results are never lost.
    """
    if not doc_ids:
        raise ValueError("no doc_ids given")
    load_credentials()
    model = load_models()[model_key]
    prompt = load_prompt(prompt_version)
    documents = corpus.load_documents(doc_ids)
    client = client or CLIENT_BUILDERS[model.provider]()
    extract = EXTRACTORS[model.provider]

    rows: list[dict] = []
    try:
        _run_documents(documents, extract, client, model, model_key, prompt, prompt_version, rows)
    finally:
        if rows:
            write_artifact(rows, model_key, prompt_version, out_dir)
            append_ledger(rows, model, prompt_version, ledger_path)
    return rows


def _run_documents(documents, extract, client, model, model_key, prompt, prompt_version, rows):
    for document in documents:
        row = new_row(
            document.ref.doc_id,
            document.ref.bank,
            document.ref.announcement_date.isoformat(),
            model_key,
            prompt_version,
        )
        try:
            raw = extract(client, model.model_id, prompt, document.as_model_input())
        except (openai.OpenAIError, anthropic.AnthropicError, EmptyResponseError) as error:
            row["error"] = f"api_error: {error}"
            # Empty responses are still billed — keep the ledger honest.
            row["input_tokens"] = getattr(error, "input_tokens", 0)
            row["output_tokens"] = getattr(error, "output_tokens", 0)
            row["cost_usd"] = round(
                model.cost_usd(row["input_tokens"], row["output_tokens"]), 6
            )
            logger.error("extraction failed for %s: %s", document.ref.doc_id, error)
            rows.append(row)
            continue
        row["input_tokens"] = raw.input_tokens
        row["output_tokens"] = raw.output_tokens
        row["cost_usd"] = round(model.cost_usd(raw.input_tokens, raw.output_tokens), 6)
        row["latency_ms"] = raw.latency_ms
        apply_payload(row, raw.payload)
        rows.append(row)
        logger.info(
            "%s: ok=%s action=%s ($%.4f, %dms)",
            document.ref.doc_id,
            row["ok"],
            (row["record"] or {}).get("action"),
            row["cost_usd"],
            row["latency_ms"],
        )


def replace_with_retries(tmp: Path, path: Path, *, attempts: int = 5) -> None:
    """Atomic swap with brief retries: os.replace is atomic, but Windows
    antivirus/indexers transiently lock the destination and throw
    PermissionError — retry instead of crashing mid-write."""
    for attempt in range(1, attempts + 1):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(0.1 * attempt)


def write_artifact(rows: list[dict], model_key: str, prompt_version: str, out_dir: Path) -> None:
    """Merge rows into the run's artifact by doc_id — never truncate prior results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_key}__{prompt_version}__{SCHEMA_VERSION}.jsonl"
    merged: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            existing = json.loads(line)
            merged[existing["doc_id"]] = existing
    for row in rows:
        merged[row["doc_id"]] = row
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in merged.values():
            handle.write(json.dumps(row) + "\n")
    replace_with_retries(tmp, path)  # atomic: a crash mid-write can't truncate paid rows
    logger.info("wrote %s (%d rows, %d updated)", path, len(merged), len(rows))


def append_ledger(
    rows: list[dict],
    model: ModelConfig,
    prompt_version: str,
    ledger_path: Path,
    *,
    run_id: str = "",
) -> None:
    """Append one spend row. ``run_id`` (the provider batch id for batch runs)
    lets collect retries detect an already-recorded batch and stay idempotent."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not ledger_path.exists()
    with ledger_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(
                ["timestamp_utc", "model_key", "prompt_version", "documents",
                 "input_tokens", "output_tokens", "cost_usd", "run_id"]
            )
        writer.writerow(
            [
                datetime.now(UTC).isoformat(timespec="seconds"),
                model.key,
                prompt_version,
                len(rows),
                sum(row["input_tokens"] for row in rows),
                sum(row["output_tokens"] for row in rows),
                round(sum(row["cost_usd"] for row in rows), 6),
                run_id,
            ]
        )
