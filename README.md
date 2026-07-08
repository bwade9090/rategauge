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

- [x] Corpus ingestion: 224 FOMC statements (2000–2026) + 302 ECB decision releases (1999–2026), fetched on demand from official indexes, nothing re-hosted ([catalog](data/catalog/documents.csv))
- [x] CBPOL-derived golden set via the BIS SDMX API, with documented derivation & exclusion rules ([data card](data/golden/README.md))
- [x] Schema-constrained extraction pipeline (OpenAI Responses API + Anthropic Messages API structured outputs; versioned prompts; JSONL artifacts + cost ledger)
- [x] Evaluation suite + CI regression gate (network-free tests via mocked models)
- [x] Batch API lifecycle at 50% token cost (idempotent submit/status/collect; crash-safe artifacts & spend ledger)
- [x] Hallucination & abstention metrics against the golden record
- [x] Model × prompt league table with Wilson 95% confidence intervals (results below)
- [ ] Trap-document set (~50 non-decision documents: minutes, speeches) for false-positive measurement
- [ ] Evaluation methodology write-up with case studies
- [ ] FastAPI service + SQLite request tracing
- [ ] Docker image & scheduled incremental ingestion

## Results — full corpus, prompt v001 (July 2026)

All 526 documents were extracted through each provider's Batch API and graded against the CBPOL-derived golden set:

| model | graded | action acc. (95% CI) | hallucination (95% CI) | bps ok | level ok | cost |
|---|---|---|---|---|---|---|
| claude-haiku-4-5 | 523/526 | 96.4% [94.4%, 97.7%] | 0.6% [0.2%, 1.7%] | 93.2% | 96.6% | $0.69 |
| gpt-5.4-nano | 475/526 | 96.2% [94.1%, 97.6%] | 2.3% [1.3%, 4.1%] | 83.2% | 93.0% | $0.10 |
| gpt-5.4-mini | 467/526 | 88.2% [85.0%, 90.8%] | 9.2% [6.9%, 12.2%] | 79.1% | 95.7% | $0.36 |

*graded* = 526 documents minus 3 that are ungradeable by construction (see the [data card](data/golden/README.md)) minus per-model extraction failures. *Hallucination* = fabricated decision ∪ wrong direction ∪ wrong change size ∪ wrong level, over graded documents. CIs are Wilson score intervals; costs are actual batch spend from the [cost ledger](eval/cost_ledger.csv).

What the harness surfaced — none of it visible without an answer key:

- **Price is not reliability.** gpt-5.4-nano is statistically indistinguishable from claude-haiku-4-5 on action accuracy (exact McNemar p = 1.0 on 475 paired documents) at a seventh of the cost — while gpt-5.4-mini, the *more expensive* OpenAI model, is significantly worse than both (38 documents that haiku gets right and mini gets wrong, zero the other way; p < 10⁻¹⁰).
- **The expensive model's failures are systematic, not noise.** gpt-5.4-mini exhibits a label inversion — `action: "hike"` alongside `change_bps: -50` and an evidence quote reading "will be reduced by 0.5 percentage points" — concentrated in the 2001 easing cycle. It abstains (`no_policy_decision`) on 2000-era Fed statement layouts the other models read fine, and it falls for the forward-guidance trap: the June 2022 ECB statement *announcing* a July hike is extracted as a June hike.
- **Evidence grounding collides with provider content filters.** 48 gpt-5.4-nano and 53 gpt-5.4-mini responses were cut off mid-generation by OpenAI's anti-regurgitation filter (`status: incomplete`, reason `content_filter`, unbilled) — triggered by the schema's `evidence_quote` field quoting official policy prose back verbatim. claude-haiku-4-5 returned 526/526 schema-valid records. For pipelines that must cite their sources, the citation requirement itself can become the availability bottleneck.
- **Abstention behaves as designed.** Effective dates are stated in ECB releases but almost never in FOMC statements; models return null rather than guess (haiku: 74 abstentions, 0 wrong answers on rate-change documents). Because the golden effective date comes from the statistical series, abstention is measured, not merely tolerated.

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
