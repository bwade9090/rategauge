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

Core pipeline complete end to end: corpus → golden set → batch extraction → grading → league table. Roadmap:

- [x] Corpus ingestion: 224 FOMC statements (2000–2026) + 303 ECB decision releases (1999–2026), fetched on demand from official indexes, nothing re-hosted ([catalog](data/catalog/documents.csv))
- [x] CBPOL-derived golden set via the BIS SDMX API, with documented derivation & exclusion rules ([data card](data/golden/README.md))
- [x] Schema-constrained extraction pipeline (OpenAI Responses API + Anthropic Messages API structured outputs; versioned prompts; JSONL artifacts + cost ledger)
- [x] Evaluation suite + CI regression gate (network-free tests via mocked models)
- [x] Batch API lifecycle at 50% token cost (idempotent submit/status/collect; crash-safe artifacts & spend ledger)
- [x] Hallucination & abstention metrics against the golden record
- [x] Model × prompt league table with Wilson 95% confidence intervals (results below)
- [x] Trap-document set (64 no-decision documents: FOMC minutes, ECB meeting accounts, non-decision releases) for false-positive measurement ([trap catalog](data/catalog/traps.csv))
- [x] Evaluation methodology write-up with case studies ([docs/eval-methodology.md](docs/eval-methodology.md))
- [x] FastAPI service + SQLite request tracing (usage below; smoke-tested in CI from the Docker image)
- [x] Docker image (`docker compose up`)
- [ ] Scheduled incremental ingestion of new decisions

## Results — full corpus + trap set, prompt v001 (July 2026)

All 527 decision documents and 64 trap documents were extracted through each provider's Batch API and graded against the CBPOL-derived golden set:

| model | graded | action acc. (95% CI) | hallucination (95% CI) | bps ok | level ok | trap FP (95% CI) | cost |
|---|---|---|---|---|---|---|---|
| claude-haiku-4-5 | 524/527 | 96.4% [94.4%, 97.7%] | 0.6% [0.2%, 1.7%] | 93.2% | 96.6% | 85.9% [75.4%, 92.4%] | $1.13 |
| gpt-5.4-nano | 476/527 | 96.2% [94.1%, 97.6%] | 2.3% [1.3%, 4.1%] | 83.2% | 93.1% | 84.4% [73.6%, 91.3%] | $0.17 |
| gpt-5.4-mini | 468/527 | 88.2% [85.0%, 90.9%] | 9.2% [6.9%, 12.2%] | 79.1% | 95.7% | 71.9% [59.9%, 81.4%] | $0.63 |

*graded* = 527 documents minus 3 that are ungradeable by construction (see the [data card](data/golden/README.md)) minus per-model extraction failures. *Hallucination* = fabricated decision ∪ wrong direction ∪ wrong change size ∪ wrong level, over graded decision documents. *Trap FP* = share of the 64 no-decision documents (FOMC minutes, ECB meeting accounts, ECB non-decision releases) on which the model fabricated a decision record instead of returning `no_policy_decision`. CIs are Wilson score intervals; costs are actual batch spend from the [cost ledger](eval/cost_ledger.csv).

What the harness surfaced — none of it visible without an answer key:

- **The headline number is not the headline risk.** The same models that hallucinate on 0.6–9.2% of real decision statements fabricate a decision record from **72–86% of documents that don't announce one** — and the model with the *best* statement hallucination score (claude-haiku, 0.6%) has the *highest* trap fabrication rate (85.9%). Shown FOMC minutes or ECB meeting accounts — documents that *recount* a decision made weeks earlier — every model extracts the recounted decision as if it were being announced (claude-haiku: 55 of 55 minutes/accounts; its evidence quote for the January 2008 minutes is the *January cut it describes*). On the nine short non-decision press releases (PEPP, TPI, TLTRO details), abstention works almost perfectly. In a production pipeline this is the difference between a clean decision table and one silently double-counting every decision three weeks later — and it is exactly the failure mode a statement-only evaluation can never see.

- **Price is not reliability.** gpt-5.4-nano is statistically indistinguishable from claude-haiku-4-5 on action accuracy (exact McNemar p = 1.0 on 476 paired documents) at a seventh of the cost — while gpt-5.4-mini, the *more expensive* OpenAI model, is significantly worse than both (38 documents that haiku gets right and mini gets wrong, zero the other way; p < 10⁻¹⁰).
- **The expensive model's failures are systematic, not noise.** gpt-5.4-mini exhibits a label inversion — `action: "hike"` alongside `change_bps: -50` and an evidence quote reading "will be reduced by 0.5 percentage point" — recurring across easing cycles from 1999 to 2025. It abstains (`no_policy_decision`) on 2000-era Fed statement layouts the other models read fine, and it falls for the forward-guidance trap: the June 2022 ECB statement *announcing* a July hike is extracted as a June hike.
- **Evidence grounding collides with provider content filters.** 48 gpt-5.4-nano and 53 gpt-5.4-mini responses were cut off mid-generation by OpenAI's anti-regurgitation filter (`status: incomplete`, reason `content_filter`, unbilled) — triggered by the schema's `evidence_quote` field quoting official policy prose back verbatim. claude-haiku-4-5 returned 527/527 schema-valid records. For pipelines that must cite their sources, the citation requirement itself can become the availability bottleneck.
- **Abstention behaves as designed.** Effective dates are stated in ECB releases but almost never in FOMC statements; models return null rather than guess (haiku: 74 abstentions, 0 wrong answers on rate-change documents). Because the golden effective date comes from the statistical series, abstention is measured, not merely tolerated.
- **The evaluation audits its own corpus.** Building the trap set exposed a real ingestion bug: the 2016-12-08 ECB decision is titled "Monetary *P*olicy *D*ecisions" (title case), and an exact-case title match had silently dropped it from the corpus. Case-insensitive matching recovered it — all three models extract the hold correctly — and the corpus is now 527 documents.

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

## Serving

`rategauge serve` (or `docker compose up`) starts a FastAPI service over the committed evaluation artifacts — the same files every number above comes from, so the API can never disagree with the published results:

```
GET  /health
GET  /decisions?model=claude-haiku-4-5&bank=ECB&date_from=2026-01-01&status=graded
GET  /eval/scorecard?model=gpt-5.4-nano
POST /extract        {"model": "gpt-5.4-nano", "doc_id": "ecb_pr121206"}
                     {"model": "gpt-5.4-nano", "bank": "FED", "text": "..."}
GET  /traces
```

`POST /extract` runs a live schema-constrained extraction (provider keys from `.env`; either a catalogued `doc_id` or raw `bank`+`text`) and traces tokens, computed USD cost, and latency to SQLite; `GET /traces` exposes the monitoring trail. Extraction failures are part of the measured domain, so they return `200` with `ok: false` — exactly like artifact rows. Interactive docs at `/docs` (OpenAPI).

## Data sources & licensing

- **Policy documents**: Federal Reserve (FOMC statements) and ECB (press releases), fetched on demand from official public sources; no document text is re-hosted in this repository.
- **Ground truth**: BIS policy-rate statistics (CBPOL) retrieved via the BIS SDMX API, used with attribution under the BIS terms and conditions.
- **Code**: MIT — see [LICENSE](LICENSE).
