# RateGauge — Technical Design

Status: accepted (2026-07-06). All external-source facts below were verified hands-on with live requests on 2026-07-06; snapshot figures are labelled accordingly.

## 1. Goal

Build a compact, production-shaped LLM application that demonstrates a full evaluation-first lifecycle:

1. **Ingest** central-bank policy decision documents (FOMC statements, ECB monetary-policy decision press releases) from official sources, on demand.
2. **Extract** structured rate-decision records with schema-constrained LLM calls.
3. **Grade** every extraction against an authoritative statistical golden set derived from the BIS central bank policy rate series (CBPOL), fetched via the BIS SDMX API — no hand-labelled ground truth.
4. **Gate** quality in CI: any prompt or model change re-runs the evaluation against a committed baseline.
5. **Serve** extractions, decision records, and evaluation scorecards from a FastAPI service with per-request token/cost/latency tracing.

**Why this task, when the extracted numbers are already published as structured data?** That redundancy is the experimental design, not an oversight. LLM document extraction is typically deployed exactly where no ground truth exists (the reason Project Gaia needed human-in-the-loop verification of extracted climate KPIs). A methodology for *measuring* extraction reliability can therefore only be validated on a task where truth is knowable — and policy decisions are the rare document-extraction task with a complete, authoritative, machine-readable answer key. The product is the measurement harness and the failure map it produces (hold-vs-change confusion, fabricated decisions on trap documents, announcement/effective date slips, era-specific wording), which transfer to extraction targets that have no API: votes, guidance, covenants, disclosures. See README "Why" for the applicant-facing version of this argument.

**Non-goals (MVP):** vote splits and dissenter names, forward-guidance classification, balance-sheet actions, banks beyond Fed/ECB, LLM-as-judge scoring, UI. These are roadmap items, not MVP.

## 2. System overview

```
            ┌────────────────────────────────────────────────────┐
            │                     sources/                       │
            │  fed.py: enumerate + fetch FOMC statements         │
            │  ecb.py: enumerate + fetch ECB decision releases   │
            │  (on-demand fetch, local cache, never re-hosted)   │
            └──────────────┬─────────────────────────────────────┘
                           │ clean document text + metadata
                           ▼
┌──────────────────┐   ┌────────────────────────┐   ┌──────────────────────────┐
│   goldenset/     │   │       extract/         │   │        evalsuite/        │
│ cbpol.py: BIS    │   │ schema-constrained LLM │   │ grader.py: join records  │
│ SDMX → daily     │──▶│ calls (sync + batch)   │──▶│ vs golden events         │
│ levels → change  │   │ prompts/ (versioned)   │   │ metrics.py: accuracy,    │
│ events (golden)  │   │ runs → JSONL artifacts │   │ hallucination, refusal,  │
└──────────────────┘   └────────────────────────┘   │ Wilson CIs, McNemar      │
                                                    └────────────┬─────────────┘
                                                                 ▼
                                            ┌────────────────────────────────┐
                                            │            serve/              │
                                            │ FastAPI: /extract /decisions   │
                                            │ /eval/scorecard  + SQLite      │
                                            │ token/cost/latency tracing     │
                                            └────────────────────────────────┘
```

Design principles:

- **Deterministic core, LLM at the edge.** Golden-set derivation and grading are pure data engineering — reproducible, unit-testable, no model in the loop.
- **Artifacts between stages.** Extraction runs write JSONL artifacts keyed by `(doc_id, model, prompt_version)`; evaluation consumes artifacts, so scoring is replayable offline and CI needs no network or API keys.
- **Ground truth behind an interface.** `GroundTruthSource` is an adapter over SDMX endpoints; BIS CBPOL is the reference implementation. ECB's own FM series slots in as a second implementation for cross-validation, and future sources (ECB SDW, IMF IFS, OECD MEI) are config, not code.
- **Institutions are config.** Each bank is a YAML manifest: document enumeration strategy, parser, golden series key, matching rules.

## 3. Data sources (verified 2026-07-06)

### 3.1 Ground truth — BIS CBPOL via SDMX

- Endpoint (confirmed): `https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/{KEY}?format=csv` with keys `D.US`, `D.XM`. Params `startPeriod`, `endPeriod`, `lastNObservations` all work. The dataflow id is `WS_CBPOL` v1.0 — legacy `WS_CBPOL_D` URLs 404.
- **Gzip is mandatory**: full-history requests stall without `Accept-Encoding: gzip` (timeout at 120 s); with gzip the full US history is ~95 KB in ~6 s. (`httpx`/`requests` send gzip by default.)
- CSV has 15 columns; `COMPILATION` contains quoted commas → real CSV parser required. Payload: `TIME_PERIOD`, `OBS_VALUE`.
- Semantics:
  - `D.US` = **midpoint of the federal funds target range** from 1985-12-19 onward (effective rate before that — irrelevant for our 2000+ window, fatal if the window is ever widened).
  - `D.XM` **redefines the tracked instrument over time** (from the series' own `COMPILATION` metadata): MRO fixed rate to 2000-06-27 → MRO minimum bid rate to 2008-10-14 → MRO fixed rate to 2024-09-17 → **deposit facility rate from 2024-09-18**. The 2024-09-18 switch produces a spurious −75 bp level shift.
- Series are forward-filled calendar-daily. Known data quirks (handled in code): `D.US` missing exactly one day (2024-10-18, value unchanged across the hole); `D.XM` has 32 weekend `NaN` observations between 2024-09-21 and 2025-01-05 → coerce and forward-fill.
- **Changes appear on the effective date, not the announcement date** (verified: FOMC 2024-09-18 announcement shifts 2024-09-19 in the series; ECB 2024-12-12 decision shifts 2024-12-18).
- Snapshot (2026-07-06): 74 US level shifts since 2000-01-01; 60 XM shifts since 1999 (including 2–3 redefinition artifacts to exclude). Latest: US midpoint 3.625 (consistent with a 3.50–3.75 % target range), XM 2.25 (DFR, after the 2026-06-17 +25 bp hike).
- Licensing: BIS statistics terms (`bis.org/terms_statistics.htm`) permit unrestricted use and republication of derived data with citation ("Source: BIS") and no implied endorsement → the derived golden-set CSV can be published in this repo with a data card.

### 3.2 Fed — FOMC statements

- Canonical path: enumerate statement URLs from official index pages — `fomccalendars.htm` (covers ~2021–present; 96 links extracted live) and `fomc_historical_year.htm` → `fomchistoricalYYYY.htm` per year for 2000–2020 (includes unscheduled meetings, e.g. 2008-01-22, 2020-03-15).
- **Never synthesize URLs from dates**: the pattern changed three times (`/boarddocs/press/monetary/...` ≤2005, `/newsevents/press/monetary/...` 2006–2010, `/newsevents/pressreleases/monetaryYYYYMMDDx.htm` current), and suffix letters are not uniform (`b` is the statement on some emergency dates). Always scrape hrefs.
- Akamai fronting: plain `curl` fails the TLS handshake from some networks; Python `httpx` with a browser-like `User-Agent` works. Include retry/backoff.
- Volume: ~224 statement events 2000-02 → 2026-06.
- Cross-check only (not canonical): HuggingFace `vtasca/fomc-statements-minutes` (465 rows, 224 statements, weekly third-party updates, known mojibake defects, imprecise `cc` license tag).
- Fed content is US-government work (public domain); we still fetch on demand and cache locally rather than re-hosting.

### 3.3 ECB — monetary policy decision press releases

- Enumeration (confirmed for every year 1999–2026): per-year HTML fragments `https://www.ecb.europa.eu/press/govcdec/mopo/{year}/html/index_include.en.html` — plain HTTP, no JS. Filter hrefs to `/press/pr/date/.../*.en.html`, dedupe across 24 languages. Snapshot total: **326 decision communications** 1999→2026-07.
- The human-facing yearly list pages and `sitemap.xml` are JS shells / section-only — unusable; the RSS feed (`/rss/press.html`) works for monitoring new releases, not backfill.
- URL styles: `pr{YYMMDD}.en.html` (≤~2013) and `ecb.mp{YYMMDD}~{hash}.en.html` (later); the hash is not constructible → enumeration only.
- Pages are ~100 KB dominated by navigation → parse the `<main>` article block only before feeding the LLM.
- **Decision date ≠ effective date** (e.g. decided 2026-06-11, effective 2026-06-17; since 2015, effect is typically the start of the next reserve-maintenance period). Press releases state the effective date explicitly → the schema carries both.
- Cross-check golden source: ECB Data Portal SDMX (`data-api.ecb.europa.eu`), event-dated series `FM.B.U2.EUR.4F.KR.{DFR|MRR_FR|MLFR}.LEV` (68/47/68 change events; verified byte-identical to the official key-rates page and to press-release text for multiple dates).
- Wording eras the prompt must handle: "minimum bid rate" during the variable-rate-tender era (2000-06 → 2008-10); corridor-only moves that leave the MRO untouched.

## 4. Golden set construction

`goldenset/cbpol.py` derives **decision events** from daily levels:

```
event = consecutive available-day pair (t-1, t) where OBS_VALUE differs
      → (ref_area, effective_date=t, old_level, new_level,
         change_bps=round((new-old)*100), direction=hike|cut)
```

Documented rules (each exclusion logged, never silently dropped):

| Rule | Rationale |
|---|---|
| Window: US ≥ 2000-01-01, XM ≥ 1999-01-01 | Pre-1986 US series is a market rate, not a target |
| US levels are range **midpoints**; extracted target ranges are graded as `(lower+upper)/2`, with the range reconstructable as midpoint ± 12.5 bp | Verified series convention |
| XM instrument regime table keyed off the verified `COMPILATION` breaks: MRO→2000-06-27, MRO min-bid→2008-10-14, MRO→2024-09-17, DFR≥2024-09-18 | Grade the extracted rate that matches the instrument CBPOL tracks on that date |
| Exclude XM redefinition artifacts (2024-09-18; audit 2000-06-28, 2008-10-15) | Series definition change, not a policy decision |
| `NaN` → forward-fill; change detection compares consecutive *available* days | 32 XM weekend NaNs; one missing US day (2024-10-18) |
| Announcement→effective matching window: Fed 0–5 days, ECB 0–10 days forward | Verified: Fed shifts land announcement+1 business day; ECB at next maintenance-period start |
| "Hold" events: documents whose announcement matches **no** golden event within the window, where the extractor must report `change_bps=0` with the prevailing level | Most FOMC/ECB communications are holds — they are the abstention test, not noise |

Sanity checks (unit-tested): expected event counts (74 US / ~57 XM after exclusions, snapshot 2026-07-06); triple-agreement spot checks vs ECB FM series and the ECB key-rates page; latest-event agreement with recent known decisions.

The derived golden set ships in the repo as `data/golden/cbpol_events.csv` + data card ("Source: BIS, Central bank policy rates (WS_CBPOL)"), alongside the code that regenerates and verifies it.

## 5. Extraction

**Schema** (Pydantic; mirrored as strict JSON Schema for both providers):

```python
class RateDecision(BaseModel):
    bank: Literal["FED", "ECB"]
    decision_date: date            # announcement date
    effective_date: date | None    # stated or inferred; None if not stated
    action: Literal["hike", "cut", "hold", "no_policy_decision"]
    change_bps: int | None         # None iff action in {hold, no_policy_decision}
    # US: target range; graded as midpoint vs CBPOL
    target_range_lower_pct: float | None
    target_range_upper_pct: float | None
    # ECB: all three key rates (the operative instrument changed over time)
    dfr_pct: float | None
    mro_pct: float | None
    mlf_pct: float | None
    evidence_quote: str            # verbatim sentence supporting the action
```

- `no_policy_decision` is the mandated abstention path for trap documents.
- Provider mechanics (both verified): OpenAI Responses API `text.format = {type: "json_schema", strict: true}`; Anthropic Messages API `output_config.format = {type: "json_schema"}`. Schema constraints honored on both: `additionalProperties: false`, no numeric min/max in the JSON Schema (range checks happen in Pydantic after parsing).
- **Prompts are versioned artifacts** (`extract/prompts/v001.md`, ...); every run records `(model_id, prompt_version, schema_version)`.
- **Batch by default, sync for iteration**: full-corpus runs go through each provider's Batch API (50 % discount, results keyed by `custom_id` — never by position); prompt iteration uses small synchronous runs on the dev subset.

**Model matrix (MVP)** — ids pinned in `models.yaml`:

| Model | Role | Full-program cost (batch, 2× margin) |
|---|---|---|
| `gpt-5.4-nano` | dev workhorse, league table | ~$0.8 |
| `gpt-5.4-mini` | league table | ~$3.0 |
| `claude-haiku-4-5-20251001` | league table (2nd provider) | ~$3.6 |
| `gpt-5.4` (stretch) | single-run quality ceiling | ~$2.9 |

Verified pricing (2026-07): gpt-5.4-mini $0.75/$4.50 per MTok, gpt-5.4-nano $0.20/$1.25, claude-haiku-4-5 $1/$5; both Batch APIs −50 %. Budget verdict: full plan ≈ $3.8 of $10 (OpenAI) + $3.6 of $5 (Anthropic) at 2× safety margin. Anthropic sync-only would blow the $5 budget → batch is mandatory there. A `cost_ledger` table records actuals per run.

## 6. Evaluation design

**Item universe:** ~550 documents (224 FOMC statements + 326 ECB decision releases), each joined to the golden set as either a change event or a hold; plus a **trap set** (~50 documents): FOMC minutes excerpts, non-monetary ECB press releases, speeches — correct output is `no_policy_decision`.

**Metrics** (per model × prompt version):

- Per-field accuracy: `action`, `change_bps` (exact), rate levels (exact at 2 dp after midpoint/regime mapping), `effective_date` (exact; window-matched fallback reported separately).
- **Hallucination rate**: share of items where the model asserts a value contradicted by the golden set (wrong direction, wrong magnitude, fabricated change on a hold), and share of trap documents where it invents a policy decision.
- **Refusal/abstention quality**: correct `no_policy_decision` on traps; incorrect abstention on real decisions.
- **Statistical layer** (the differentiating rigor): Wilson score 95 % CIs on every headline rate (percentile-bootstrap intervals were rejected — they collapse to zero width whenever the observed rate is exactly 0 or 1, publishing false certainty); McNemar exact tests for model-vs-model differences on paired items; results reported per bank and per era (URL-pattern eras double as text-style eras).
- **Event-document assignment**: each golden event belongs to exactly one document — the one with the latest announcement date at or before the effective date. Without this rule, an intermeeting cut (e.g. the post-9/11 2001-09-17 ECB emergency cut, effective 09-18) also falls inside the preceding scheduled hold's window and would invert that document's grade.
- **Known ungradeable documents** (flagged, never silently graded): the 2024-09-12 ECB decision (its effective date coincides with the CBPOL MRO→DFR series redefinition) and the corridor-only decisions 2015-12-03 / 2019-09-12 (DFR moved, the MRO that CBPOL then tracked did not). Grading those against the ECB FM deposit-facility series is a roadmap item. The 2008-12-16 Fed point-target→range transition grades on action and level but not change_bps (the golden −88 bp is a midpoint-convention artifact no document can state).

**Baseline & regression gate:** each accepted run commits a scorecard JSON under `eval/baselines/`. CI recomputes metrics from committed artifacts (offline, no keys) and fails if any headline metric degrades beyond a tolerance vs baseline. Framework note: grading is deterministic joining, so the core harness is plain pytest + pandas; packaging the trap-set eval as an `inspect-ai` task is a stretch item for ecosystem compatibility, not an MVP dependency.

## 7. Serving & tracing

FastAPI app (`serve/api.py`):

- `POST /extract` — fetch (or accept) a document, run extraction with the configured model, return the record + trace id.
- `GET /decisions` — the graded decision dataset (filter by bank/date).
- `GET /eval/scorecard` — latest scorecard(s) per model × prompt version.
- `GET /health`.

Every LLM call writes to SQLite (`traces` table): timestamp, doc id, model, prompt version, input/output tokens, computed USD cost, latency ms, outcome. `Dockerfile` + `docker compose` for one-command startup; a scheduled GitHub Actions workflow (later) performs incremental ingestion of new decisions.

## 8. Testing & CI

- **All tests network-free.** Committed fixtures: small real excerpts of CBPOL/ECB CSV responses and document HTML (US-gov public domain; BIS/ECB permit quotation with attribution), plus recorded LLM responses as JSON. LLM clients are faked at the transport layer.
- Unit tests: golden-set derivation (event counts, regime mapping, NaN/missing-day handling), document parsers per era, schema validation, grader matching rules, metrics math.
- GitHub Actions: `ruff check` + `pytest` on push/PR (Python 3.12–3.13 matrix, matching the pinned dependency set); the eval regression gate runs on the committed artifacts.

## 9. Repository layout

```
src/rategauge/
  config.py            # pydantic-settings; .env for keys
  schema.py            # RateDecision + JSON Schema export
  sources/             # fed.py, ecb.py, cache.py
  goldenset/           # cbpol.py, ecb_fm.py (cross-check), builder.py
  extract/             # clients (openai/anthropic, sync+batch), runner.py, prompts/
  evalsuite/           # grader.py, metrics.py, scorecard.py
  serve/               # api.py, tracing.py
  cli.py               # rategauge ingest|golden|extract|eval|serve
configs/               # banks.yaml, models.yaml
data/                  # gitignored cache; data/golden/ + data/catalog/ committed (derived facts, licensed)
eval/baselines/        # committed scorecards
tests/                 # incl. fixtures/
docs/                  # DESIGN.md, eval-methodology.md (written with results)
```

## 10. Milestones

| Milestone | Target | Definition of done |
|---|---|---|
| M1 Golden set + ingestion | Jul 12 | CBPOL golden set derived & sanity-tested; Fed + ECB documents enumerated and cached; matching rules validated on 2024–2026 events |
| M2 Extraction + eval | Jul 19 | 3-model × prompt-version league table with CIs on the full corpus + trap set; CI regression gate green |
| M3 Serve + ship | Jul 22 | FastAPI + tracing + Docker; eval-methodology doc; README with headline results |

Fallback (pre-agreed): if the CBPOL join fights back past Jul 12, ship Fed-only and cut ECB — the Fed-only pipeline is a complete story.

## 11. Licensing & data ethics

- Code: MIT. Documents: fetched from official sources at runtime, cached locally, never re-hosted.
- BIS-derived golden set: published with citation "Source: BIS, Central bank policy rates (WS_CBPOL)", no implied endorsement, no added charge.
- Polite crawling: cache-first, backoff, identifiable User-Agent; ~550 documents once, then incremental.
