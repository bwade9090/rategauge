"""FastAPI service: extractions, decision records, and evaluation scorecards.

Read endpoints serve the committed evaluation artifacts — the same files every
README number comes from — so the API can never disagree with the published
results. ``POST /extract`` runs a live schema-constrained extraction and
traces token/cost/latency to SQLite (``GET /traces`` for monitoring).

Start with ``rategauge serve`` (or ``uvicorn rategauge.serve.api:app``) from
the repo root: paths to artifacts, catalogs, and ``.env`` are relative.
"""

import json
import logging
import os
from datetime import UTC, date, datetime
from pathlib import Path

import anthropic
import openai
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from rategauge import corpus
from rategauge.config import load_credentials, load_models
from rategauge.extract.clients import CLIENT_BUILDERS, EXTRACTORS, EmptyResponseError
from rategauge.extract.runner import RUNS_DIR, load_prompt
from rategauge.schema import SCHEMA_VERSION, RateDecision
from rategauge.serve import tracing
from rategauge.sources.common import DocumentRef

logger = logging.getLogger(__name__)

GRADED_DIR = Path("eval/graded")
SCORECARDS_DIR = Path("eval/scorecards")
BANKS = ("FED", "ECB")
# model/prompt_version become filename components; no path separators, ever.
SAFE_IDENTIFIER = r"^[A-Za-z0-9._-]+$"


class ExtractRequest(BaseModel):
    """Either a catalogued doc_id, or a raw document (bank + text)."""

    model: str
    prompt_version: str = Field(default="v001", pattern=SAFE_IDENTIFIER)
    doc_id: str | None = None
    bank: str | None = None
    text: str | None = Field(default=None, max_length=200_000)


def _default_traces_path() -> Path:
    # Overridable so containers can point the trace DB at a mounted volume.
    return Path(os.environ.get("RATEGAUGE_TRACES_DB", str(tracing.TRACES_PATH)))


def create_app(
    *,
    graded_dir: Path = GRADED_DIR,
    scorecards_dir: Path = SCORECARDS_DIR,
    runs_dir: Path = RUNS_DIR,
    traces_path: Path | None = None,
) -> FastAPI:
    traces_path = traces_path if traces_path is not None else _default_traces_path()
    app = FastAPI(
        title="RateGauge",
        description=(
            "Evaluation-first LLM extraction of monetary policy decisions, "
            "auto-graded against official central-bank statistics (BIS CBPOL)."
        ),
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "schema_version": SCHEMA_VERSION}

    @app.get("/decisions")
    def decisions(
        model: str = Query(pattern=SAFE_IDENTIFIER),
        prompt_version: str = Query(default="v001", pattern=SAFE_IDENTIFIER),
        bank: str | None = Query(default=None, pattern="^(FED|ECB)$"),
        date_from: date | None = None,
        date_to: date | None = None,
        status: str | None = None,
        limit: int = Query(default=500, ge=1, le=2000),
    ) -> dict:
        """The graded decision dataset for one model x prompt, with the
        extracted record joined in; filter by bank, date range, or status."""
        stem = f"{model}__{prompt_version}__{SCHEMA_VERSION}"
        graded_path = graded_dir / f"{stem}.jsonl"
        if not graded_path.exists():
            raise HTTPException(404, f"no graded artifact for {model}/{prompt_version}")
        records: dict[str, dict | None] = {}
        runs_path = runs_dir / f"{stem}.jsonl"
        if runs_path.exists():
            for line in runs_path.read_text(encoding="utf-8").splitlines():
                if line:
                    run_row = json.loads(line)
                    records[run_row["doc_id"]] = run_row.get("record")
        rows = []
        for line in graded_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            announced = date.fromisoformat(row["announcement_date"])
            if bank and row["bank"] != bank:
                continue
            if date_from and announced < date_from:
                continue
            if date_to and announced > date_to:
                continue
            if status and row["status"] != status:
                continue
            row["record"] = records.get(row["doc_id"])
            rows.append(row)
            if len(rows) >= limit:
                break
        return {"count": len(rows), "rows": rows}

    @app.get("/eval/scorecard")
    def scorecard(model: str | None = None) -> dict:
        """Latest committed scorecard per model x prompt version."""
        cards = []
        for path in sorted(scorecards_dir.glob("*.json")):
            card = json.loads(path.read_text(encoding="utf-8"))
            if model and card.get("model_key") != model:
                continue
            cards.append(card)
        if model and not cards:
            raise HTTPException(404, f"no scorecard for model {model}")
        return {"count": len(cards), "scorecards": cards}

    @app.get("/traces")
    def traces(limit: int = Query(default=50, ge=1, le=500)) -> dict:
        """Most recent live-extraction traces (token/cost/latency monitoring)."""
        return {"traces": tracing.recent_traces(traces_path, limit=limit)}

    @app.post("/extract")
    def extract(request: ExtractRequest) -> dict:
        """Run one live schema-constrained extraction and trace it.

        Extraction failures (provider errors, schema violations) are part of
        the measured domain, so they return 200 with ``ok: false`` — exactly
        like artifact rows. Only caller mistakes are HTTP errors.
        """
        models = load_models()
        if request.model not in models:
            raise HTTPException(422, f"unknown model; available: {sorted(models)}")
        model = models[request.model]
        try:
            prompt = load_prompt(request.prompt_version)
        except FileNotFoundError as error:
            raise HTTPException(422, str(error)) from error
        document = _resolve_document(request)

        load_credentials()
        client = CLIENT_BUILDERS[model.provider]()
        usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "latency_ms": 0}
        ok, record, error = False, None, None

        def trace(outcome_ok: bool, outcome_error: str | None) -> int | None:
            return _record_trace_safely(
                traces_path,
                doc_id=document.ref.doc_id,
                bank=document.ref.bank,
                model_key=request.model,
                prompt_version=request.prompt_version,
                ok=outcome_ok,
                error=outcome_error,
                **usage,
            )

        try:
            raw = EXTRACTORS[model.provider](
                client, model.model_id, prompt, document.as_model_input()
            )
        except (openai.OpenAIError, anthropic.AnthropicError, EmptyResponseError) as exc:
            error = f"api_error: {exc}"
            # Empty responses are still billed — record what the provider charged.
            usage["input_tokens"] = getattr(exc, "input_tokens", 0)
            usage["output_tokens"] = getattr(exc, "output_tokens", 0)
            usage["cost_usd"] = round(
                model.cost_usd(usage["input_tokens"], usage["output_tokens"]), 6
            )
            logger.error("live extraction failed for %s: %s", document.ref.doc_id, exc)
        except Exception as exc:
            # Even unexpected failures after a (possibly billed) round-trip
            # must leave a trace before surfacing as a 500.
            trace(False, f"unexpected: {exc}")
            raise
        else:
            usage = {
                "input_tokens": raw.input_tokens,
                "output_tokens": raw.output_tokens,
                "cost_usd": round(model.cost_usd(raw.input_tokens, raw.output_tokens), 6),
                "latency_ms": raw.latency_ms,
            }
            try:
                record = json.loads(
                    RateDecision.model_validate_json(raw.payload).model_dump_json()
                )
                ok = True
            except ValueError as exc:  # covers json + pydantic validation errors
                error = f"schema_violation: {exc}"

        trace_id = trace(ok, error)
        return {"ok": ok, "record": record, "error": error, "trace_id": trace_id,
                "usage": usage}

    return app


def _record_trace_safely(traces_path: Path, **kwargs) -> int | None:
    """A paid extraction result must reach the caller even if the trace DB is
    unwritable — log the tracing failure, return trace_id None."""
    try:
        return tracing.record_trace(traces_path, **kwargs)
    except Exception:
        logger.exception("failed to write trace to %s", traces_path)
        return None


def _resolve_document(request: ExtractRequest) -> corpus.LoadedDocument:
    if request.doc_id:
        try:
            # fetch_missing: DESIGN §7 says "fetch (or accept) a document" —
            # a fresh container has a catalog but an empty cache.
            [document] = corpus.load_documents([request.doc_id], fetch_missing=True)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(502, f"could not fetch document: {error}") from error
        return document
    if not request.text or request.bank not in BANKS:
        raise HTTPException(
            422, "provide either doc_id, or bank (FED|ECB) together with text"
        )
    # Ad-hoc document: same provenance framing as catalogued ones, so the
    # measured prompt contract is identical.
    ref = DocumentRef(
        bank=request.bank,
        doc_id="adhoc",
        announcement_date=datetime.now(UTC).date(),
        url="",
        doc_type="statement",
    )
    return corpus.LoadedDocument(ref=ref, text=request.text)


app = create_app()
