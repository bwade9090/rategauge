# CBPOL-derived golden set

`cbpol_events.csv` contains policy-rate **decision events** derived from the BIS
central bank policy rate statistics (daily series, jurisdictions `US` and `XM`),
regenerated at any time with:

```
python -m rategauge.goldenset.cbpol --out data/golden
```

**Source: BIS, Central bank policy rates (WS_CBPOL), https://data.bis.org — © Bank
for International Settlements.** Derived and republished under the [BIS terms for
statistics](https://www.bis.org/terms_statistics.htm) with attribution; no
endorsement by the BIS is implied.

## Columns

| column | meaning |
|---|---|
| `ref_area` | `US` (federal funds target range) or `XM` (euro area) |
| `effective_date` | date the new level appears in the daily series (**effective**, not announcement, date) |
| `old_level` / `new_level` | policy-rate level in percent; US levels are target-range **midpoints** |
| `change_bps` | `round((new − old) × 100)` |
| `direction` | `hike` / `cut` |
| `instrument` | `FFTR_MID` (US), `MRO` or `DFR` (XM, per the series' regime table) |
| `excluded` | `True` = detected level shift that is **not** a golden decision event |
| `exclusion_reason` | why (outside window / series redefinition) — exclusions are kept, never silently dropped |
| `audit_note` | flag for events coinciding with XM tender-procedure changes (cross-checked against ECB sources) |

## Snapshot (generated 2026-07-06)

- `US`: 74 golden events, 2000-02-02 → 2025-12-11 (+3 pre-window 1999 shifts, excluded)
- `XM`: 59 golden events, 1999-04-09 → 2026-06-17 (+1 excluded: the 2024-09-18
  MRO→DFR series redefinition, an apparent −75 bp shift that is not a policy decision)

See [docs/DESIGN.md](../../docs/DESIGN.md) §3.1 and §4 for the verified series
conventions and the full derivation rules.
