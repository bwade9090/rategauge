# RateGauge

**Evaluation-first LLM extraction of monetary policy decisions — auto-graded against official central-bank statistics.**

RateGauge turns unstructured central-bank policy statements (FOMC, ECB) into structured, machine-readable rate-decision records using LLMs with schema-constrained outputs — and then measures its own reliability by grading every extraction against the official BIS policy-rate statistics (CBPOL), fetched via the SDMX API. No hand-labelled golden set: the ground truth is the statistical record itself.

> Trust in an LLM pipeline shouldn't be a vibe. It should be a number, computed in CI, against an authoritative source.

## Why

Central banks and financial institutions increasingly use LLMs to extract structured data from unstructured documents (see, e.g., the BIS Innovation Hub's Project Gaia, which extracts climate indicators from corporate reports). The hard part is not extraction — it is knowing when the model is wrong. And in most real deployments there is no ground truth to check against: that is precisely why LLM extraction is used in the first place, and why its reliability is so hard to certify.

RateGauge inverts that trap deliberately: it runs a full production-shaped extraction pipeline on a task where a perfect audit trail exists — monetary-policy decisions, recorded authoritatively in the BIS's official CBPOL statistics — so that every hallucination is countable, at zero labelling cost.

### Why grade a task an official API already answers?

Because the deliverable is not the data — it is the **measured trustworthiness of the method**. Nobody needs an LLM to learn the current federal funds target range. The value of re-deriving it from prose is that the derivation can be scored *exactly*, the same way a nowcasting model is backtested on periods where the outcome is already known before anyone trusts it in real time. Policy statements are structurally identical to the documents that matter in production — official financial communications, with numbers, dates, and hedged language embedded in careful prose whose conventions shift across eras — but uniquely among such documents, they come with a complete statistical answer key.

What the evaluation produces is a transferable **failure map**: how often models misread a "hold" as a cut, invent decisions from documents that contain none, confuse announcement dates with effective dates, or stumble over era-specific wording ("minimum bid rate", target ranges vs single rates) — quantified with confidence intervals, per model and per prompt version, and enforced as a regression gate in CI. Those are exactly the failure modes that matter when the same technique is pointed at documents with **no** answer key: vote splits and dissents, forward guidance, loan covenants, climate disclosures.

RateGauge is a compact, production-shaped reference implementation of that idea:

1. **Extract** — LLM + Pydantic schema-constrained outputs turn policy statements into decision records (rate level, change in basis points, direction, effective date).
2. **Grade** — extractions are joined against the official BIS CBPOL policy-rate series (SDMX) to produce per-field accuracy, hallucination, and refusal rates.
3. **Gate** — the evaluation suite runs in CI as a regression gate on every prompt or model change.
4. **Serve** — a FastAPI service exposes extractions, decision records, and live scorecards with token/cost/latency tracing.

## Status

Early development (project scaffold). Roadmap:

- [x] Corpus ingestion: 224 FOMC statements (2000–2026) + 302 ECB decision releases (1999–2026), fetched on demand from official indexes, nothing re-hosted ([catalog](data/catalog/documents.csv))
- [x] CBPOL-derived golden set via the BIS SDMX API, with documented derivation & exclusion rules ([data card](data/golden/README.md))
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
