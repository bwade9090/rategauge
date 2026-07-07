# Document catalog

`documents.csv` is the enumerated corpus: one row per official monetary-policy
document (facts only — bank, id, announcement date, source URL, type). It is
committed for reproducibility; the document *texts* are never re-hosted — they
are fetched on demand from the official sources into the gitignored
`data/cache/` directory by:

```
rategauge ingest
```

Enumeration paths (verified live, 2026-07; see [docs/DESIGN.md](../../docs/DESIGN.md) §3.2–3.3):

- **FED** — statement links scraped from the official FOMC calendar page
  (~2021+) and per-year historical pages (2000–2020); URLs are never
  synthesized from dates (the URL scheme changed three times, and letter
  suffixes are irregular on emergency dates).
- **ECB** — 'Monetary policy decisions' releases from the per-year
  `press/govcdec/mopo/{year}` HTML fragments (1999+), English only,
  accounts/PDF/language duplicates excluded.
