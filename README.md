# RateGauge

**Evaluation-first LLM extraction of monetary policy decisions — auto-graded against official central-bank statistics.**

RateGauge turns unstructured central-bank policy statements (FOMC, ECB) into structured, machine-readable rate-decision records using LLMs with schema-constrained outputs — and then measures its own reliability by grading every extraction against the official BIS policy-rate statistics (CBPOL), fetched via the SDMX API. No hand-labelled golden set: the ground truth is the statistical record itself.

> Trust in an LLM pipeline shouldn't be a vibe. It should be a number, computed in CI, against an authoritative source.

## Why

Central banks and financial institutions increasingly use LLMs to extract structured data from unstructured documents (see, e.g., the BIS Innovation Hub's Project Gaia). The hard part is not extraction — it is knowing when the model is wrong. RateGauge is a compact, production-shaped reference implementation of one answer:

1. **Extract** — LLM + Pydantic schema-constrained outputs turn policy statements into decision records (rate level, change in basis points, direction, effective date).
2. **Grade** — extractions are joined against the official BIS CBPOL policy-rate series (SDMX) to produce per-field accuracy, hallucination, and refusal rates.
3. **Gate** — the evaluation suite runs in CI as a regression gate on every prompt or model change.
4. **Serve** — a FastAPI service exposes extractions, decision records, and live scorecards with token/cost/latency tracing.

## Status

Early development (project scaffold). Roadmap:

- [ ] Corpus ingestion: FOMC statements + ECB press releases (fetched on demand, nothing re-hosted)
- [ ] CBPOL-derived golden set via the BIS SDMX API, with documented matching rules
- [ ] Schema-constrained extraction pipeline
- [ ] Evaluation suite + CI regression gate (network-free tests via mocked models)
- [ ] Hallucination / refusal metrics on control and trap documents
- [ ] Model × prompt league table with bootstrap confidence intervals
- [ ] FastAPI service + SQLite request tracing
- [ ] Docker image & scheduled incremental ingestion

## Quickstart (development)

```powershell
# Windows
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e . -r requirements-dev.txt
pytest
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e . -r requirements-dev.txt
pytest
```

## Data sources & licensing

- **Policy documents**: Federal Reserve (FOMC statements) and ECB (press releases), fetched on demand from official public sources; no document text is re-hosted in this repository.
- **Ground truth**: BIS policy-rate statistics (CBPOL) retrieved via the BIS SDMX API, used with attribution under the BIS terms and conditions.
- **Code**: MIT — see [LICENSE](LICENSE).
