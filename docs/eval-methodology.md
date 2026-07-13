# Evaluation Methodology

How RateGauge measures the trustworthiness of LLM extraction — and what the
measurements revealed. Every number in this document is recomputable from the
committed artifacts with `rategauge grade` (see [Reproducibility](#reproducibility)).

## 1. The design principle: evaluate where truth is knowable

LLM extraction is normally deployed exactly where no ground truth exists —
that is why it is used. The consequence is that its reliability is asserted,
not measured. RateGauge inverts the trap: it runs a production-shaped
extraction pipeline on a task with a perfect audit trail (monetary-policy
decisions, recorded authoritatively in the BIS CBPOL statistics), so that
every hallucination is countable at zero labelling cost. The analogy is
backtesting: a nowcasting model earns trust on periods where the outcome is
already known before anyone uses it in real time.

The deliverable is therefore not the extracted data — official APIs already
publish it — but a **transferable failure map**: quantified, per-model,
per-prompt evidence of *how* extraction goes wrong on official financial
prose. Sections 6–7 show that the map's most important entries would be
invisible to any evaluation that only measures accuracy on well-behaved
inputs.

## 2. Ground truth: a golden set derived, not labelled

The golden set ([data card](../data/golden/README.md)) is derived from the
BIS policy-rate series (CBPOL, daily, via SDMX): 74 US and 59 euro-area rate
*change events*, each with an effective date, direction, size, and level. Key
derivation rules, all documented in [DESIGN.md](DESIGN.md):

- **Series semantics are not uniform.** The US series records target-range
  *midpoints*; euro-area levels track the MRO until 2024-09-17 and the deposit
  facility rate afterwards (an instrument redefinition, not a policy move).
  Grading respects the instrument regime in force on each date.
- **Changes land on effective dates, not announcement dates.** A Fed decision
  announced Wednesday appears in the series on Thursday; ECB decisions take
  effect at dates stated in the release (historically the next maintenance
  period). Grading joins documents to events through forward windows
  (Fed 0–5 days, ECB 0–10 days).
- **Exclusions are kept, with reasons, never dropped.** Four detected shifts
  are excluded from grading (the 2024-09-18 MRO→DFR redefinition among them)
  but retained in the file — and still used for level reconstruction, so
  prevailing-level grading stays correct across the redefinition.

Three categories of documents are *ungradeable by construction* and are
flagged, never silently graded: documents whose window contains an excluded
redefinition shift; the two corridor-only ECB decisions (2015-12-03,
2019-09-12: the deposit rate moved while the tracked MRO did not); and the
2008-12-16 change size (the golden −88 bp is a midpoint-convention artifact
no document can state — its action and level remain gradeable).

## 3. The event-ownership rule (a case study in silent grade inversion)

A naive window join produces a subtle, catastrophic error. The ECB's
post-9/11 emergency cut was announced on 2001-09-17 and effective 2001-09-18.
The *scheduled* meeting of 2001-09-13 — a hold — also has the 18th inside its
10-day window. Under a naive join, a model that correctly extracted the
13th's hold would be graded **wrong** (and flagged as hallucinating), while a
model that *fabricated a cut* on the 13th would be graded **perfect**.

The fix is the ownership rule: an event belongs to a document only if no
later document was announced by the event's effective date. It is one
`bisect` comparison ([grader.py](../src/rategauge/evalsuite/grader.py)), it
is pinned by a real-corpus regression test, and it was found not by
inspection of outputs but by adversarial review of the grader itself. The
general lesson: **the evaluation harness needs evaluating**. Every grading
rule in this repo was reviewed adversarially before its numbers were
published, and several published-number bugs were caught that way (see §8).

## 4. What is measured, and why abstention is a first-class outcome

Every graded field separates three outcomes: **correct**, **wrong** (a value
was asserted and contradicts the golden set), and **abstained** (null). The
prompt demands null for anything the document does not state, so abstention
is instructed behavior — and it must be *measured*, not merely tolerated,
because the two failure directions are asymmetric: a wrong value poisons a
data table silently; a null is visible and recoverable.

- **Hallucination** := fabricated decision ∪ wrong direction ∪ wrong change
  size ∪ wrong level, over graded decision documents. Abstentions never count.
- **decision_date** is graded against the catalog announcement date — ground
  truth that comes free with enumeration.
- **effective_date** shows why the three-way split matters: on rate-change
  documents claude-haiku-4-5 scores 59 correct / 74 abstained / 0 wrong.
  A binary "accuracy 44%" reading would call this failure; the split shows a
  model that answers when the document states the date (ECB releases) and
  correctly refuses when it does not (FOMC statements).

The flagship abstention case is the 2020-03-15 emergency statement, which
does **not** state the cut size. gpt-5.4-nano fabricated a size — −150 bp in
the committed artifact, −50 bp in an earlier development run (the truth is
−100; that the fabricated value changes between runs is itself the finding).
claude-haiku-4-5 answered −100 bp: factually right, but the document doesn't
say it — the value leaked from the model's world knowledge, violating the
evidence contract. Golden grading catches the first failure; only
evidence-grounding/abstention measurement catches the second. An evaluation
needs both.

## 5. Statistics: chosen for behavior at the boundaries

- **Wilson score intervals** for every published rate. Percentile bootstrap
  was implemented first and rejected: at observed rates of exactly 0 or 1 it
  degenerates to a zero-width interval — 11/11 correct would publish
  "100% [100%, 100%]" where the honest interval is [74.1%, 100%]. Small-n
  eval tables live at those boundaries, so this is not a corner case.
- **Exact McNemar tests** for model-vs-model differences: both models grade
  the same documents, so the paired test on discordant pairs is the correct
  comparison — and it is what turned "88.2% vs 96.4% looks worse" into
  "38 documents haiku gets right and mini gets wrong, zero the reverse,
  p ≈ 7×10⁻¹² (n = 468 paired)". On the 12-document development set that
  preceded the full runs, nano vs haiku gave p = 1.0 with intervals spanning
  more than 25 points — the small-n CIs correctly warned that only the full
  corpus could decide model differences.
- **One denominator per column.** The trap false-positive rate is computed
  over all 64 trap documents for every model (an extraction failure produced
  no record, hence no fabrication). An earlier draft divided by graded traps
  only, giving each model a different denominator (64/61/59) — flagged by
  adversarial review before publication and fixed.

## 6. Failure map I: decision documents

Full corpus (527 documents: 224 FOMC statements 2000–2026, 303 ECB decision
releases 1999–2026), prompt v001, batch runs. Headline table in the
[README](../README.md); the qualitative findings:

- **Price is not reliability.** gpt-5.4-nano ≈ claude-haiku-4-5 on action
  accuracy (p = 1.0) at a seventh of the cost; gpt-5.4-mini — the more
  expensive OpenAI model — is significantly worse than both.
- **Systematic, not random.** mini's errors have shapes: a label inversion —
  `action: "hike"` alongside `change_bps: -50` and an evidence quote reading
  "will be reduced by 0.5 percentage point" (April 1999) — recurring across
  easing cycles from 1999 to 2025; mass abstention on 2000-era Fed statement
  layouts the other models read fine (8 of 8 fed-2000 documents returned
  `no_policy_decision`); and the forward-guidance trap (the June 2022 ECB
  statement *announcing* a July hike, extracted as a June hike). Era-specific
  formatting and hedged forward language are exactly the properties of
  official financial prose that transfer to other document classes.
- **The citation requirement is an availability risk.** 48 gpt-5.4-nano and
  53 gpt-5.4-mini responses were killed mid-generation by OpenAI's
  anti-regurgitation content filter (`status: incomplete`, unbilled) —
  triggered by the schema's `evidence_quote` field quoting policy prose back
  verbatim, disproportionately on long QE-era statements. claude-haiku-4-5:
  527/527 valid. A pipeline that must cite its sources can lose ~10%
  availability to the citation itself, and which documents it loses is
  provider-dependent.

## 7. Failure map II: the trap set, and the headline number

The trap set measures false positives: 64 official documents that announce
**no** policy-rate decision — 25 FOMC minutes (first scheduled meeting per
year, 2000–2024), 30 ECB monetary-policy meeting accounts, and 9 ECB
non-decision monetary-policy releases (PEPP/TPI/TLTRO details), audited to
contain no key-rate decisions ([trap catalog](../data/catalog/traps.csv)).
The only correct extraction is `no_policy_decision`; the prompt names meeting
minutes explicitly as the abstention case.

Result: models fabricate a decision record from **72–86%** of trap documents
— and the model with the *best* hallucination score on real statements
(claude-haiku-4-5, 0.6%) has the *highest* trap fabrication rate (85.9%).
The structure is sharp:

| trap type | behavior |
|---|---|
| 9 short non-decision releases | near-perfect abstention (haiku 9/9, mini 9/9, nano 7/9) |
| 55 minutes & meeting accounts | near-total fabrication (haiku 55/55) |

The models extract the decision the document *recounts*: haiku's evidence
quote for the January 2008 minutes is the January cut those minutes describe,
returned as `action: "cut", change_bps: -50` three weeks after the fact. In a
production decision-record pipeline this double-counts every decision at
minutes-release time — with correct-looking values and verbatim evidence
quotes, which is precisely why no accuracy metric on decision documents can
see it. Abstention capability is not the bottleneck (the nine releases prove
it); the failure is triggered by *recounted* decisions, a distinction the
prompt states and the models override. That makes it a prompt-and-guardrail
problem, and the harness is the regression gate for fixing it: any v002
prompt must move trap FP down without moving statement accuracy.

## 8. The evaluation audits its own inputs

Two corpus defects were found by the evaluation work itself, not by corpus
QA:

- **The 2016-12-08 ECB decision was missing.** Its index title is "Monetary
  *P*olicy *D*ecisions" (title case); an exact-case title match had silently
  dropped it. Found while enumerating trap candidates ("everything that is
  *not* titled Monetary policy decisions"), fixed with case-insensitive
  matching, and verified: all three models extract the recovered hold
  correctly. Silent-loss guards (yearly coverage floors, loud failures on
  unknown documents) exist precisely because enumeration rules rot.
- **Trap/control separation is enforced, loudly.** Trap documents live in a
  separate catalog so their dates can never enter event-ownership resolution
  (minutes are released weeks after the meetings they describe). Grading
  fails with an explicit error if an artifact contains a document in neither
  catalog — because the alternative, discovered in review by simulation, was
  64 trap rows silently graded as controls, moving published accuracy by
  5 points with no warning.

## 9. Reproducibility

- Extraction artifacts (JSONL, one row per document × model × prompt),
  graded rows, scorecards, and the cost ledger are committed under
  [eval/](../eval/). `rategauge grade --models ...` regenerates every number.
- Runs are identified by (model, prompt version, schema version); artifacts
  merge by document id, so partial re-runs never lose paid-for rows.
- Batch runs (50% token pricing) use an idempotent submit/status/collect
  lifecycle with crash-safe writes; collect retries cannot double-count spend
  (the ledger carries a `run_id`).
- Documents are fetched on demand from official sources and never re-hosted;
  the committed catalogs contain facts only (ids, dates, URLs). Total LLM
  spend for everything in this document: **≈ $2.01**
  ([cost ledger](../eval/cost_ledger.csv)).

## 10. Limitations and transfer

- Two banks, English documents, one prompt version, three models: the map is
  a foundation, not a census. The harness is built to grow along exactly
  those axes (banks-as-config, versioned prompts, model registry).
- CBPOL grades *rate decisions* only. Corridor-only decisions need the ECB's
  FM series (roadmap); non-rate decisions (QE volumes, guidance) have no
  golden source here — they inherit the trap-set treatment instead.
- The trap finding suggests the highest-value next measurement for any
  extraction deployment: not "how accurate is the model on the documents you
  meant to feed it" but "what does it do on the adjacent documents it will
  inevitably be fed". For central-bank text — vote splits, dissents, forward
  guidance, covenants, disclosures — those adjacents are the corpus.
