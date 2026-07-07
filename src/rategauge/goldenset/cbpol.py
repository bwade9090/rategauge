"""Golden-set derivation from the BIS central bank policy rate statistics (CBPOL).

The BIS publishes daily policy-rate levels per jurisdiction via the SDMX v2 API
(dataflow ``WS_CBPOL`` v1.0). RateGauge derives *decision events* from level
shifts between consecutive available observations and uses them as the golden
set that LLM extractions are graded against.

Series conventions (verified against the live API on 2026-07-06; see
docs/DESIGN.md section 3.1):

- ``D.US`` is the midpoint of the federal funds target range from 1985-12-19
  onward (the effective market rate before that, which is why the US window
  starts at 2000-01-01: pre-1986 "shifts" would be market noise, not decisions).
- ``D.XM`` redefines the tracked instrument over time — MRO variants until
  2024-09-17, the deposit facility rate afterwards. The level shift on
  2024-09-18 is a series redefinition, not a policy decision, and is excluded.
- Levels are forward-filled calendar-daily, but the series contain occasional
  missing days and ``NaN`` observations, so shifts are computed between
  consecutive *available* observations, never assuming calendar continuity.
- Shifts land on the decision's EFFECTIVE date, not its announcement date.

Every detected shift is kept in the output with an ``excluded`` flag and a
reason — documented exclusion over silent dropping.
"""

import argparse
import io
import logging
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

from rategauge.http import default_client

logger = logging.getLogger(__name__)

CBPOL_URL_TEMPLATE = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/D.{ref_area}"
REF_AREAS = ("US", "XM")

# Fetch a little before each window so the first in-window shift has a
# previous observation to diff against.
DEFAULT_START_PERIOD = "1998-12-01"

# Events strictly before these dates are flagged as outside the project window.
WINDOW_START = {"US": date(2000, 1, 1), "XM": date(1999, 1, 1)}

US_INSTRUMENT = "FFTR_MID"  # federal funds target range, midpoint convention

# XM instrument regimes, from the series' own COMPILATION metadata.
# (regime start date, instrument); a regime runs until the next entry's start.
XM_REGIMES: tuple[tuple[date, str], ...] = (
    (date(1999, 1, 1), "MRO"),  # fixed rate tender
    (date(2000, 6, 28), "MRO"),  # variable rate tender, minimum bid rate
    (date(2008, 10, 15), "MRO"),  # fixed rate tender again
    (date(2024, 9, 18), "DFR"),  # deposit facility rate
)

# 2024-09-18 switches the tracked instrument (MRO -> DFR): the -75 bp level
# shift on that date is an artifact of the redefinition, not a decision.
XM_REDEFINITION_DATES = frozenset({date(2024, 9, 18)})

# These boundaries changed the MRO tender procedure without switching
# instrument; a shift there can still be a genuine decision (2008-10-15 is:
# the coordinated -50 bp cut). Kept, but flagged for cross-source audit.
XM_AUDIT_DATES = frozenset({date(2000, 6, 28), date(2008, 10, 15)})


@dataclass(frozen=True)
class RateEvent:
    """One detected policy-rate level shift, graded golden if not excluded."""

    ref_area: str
    effective_date: date
    old_level: float
    new_level: float
    change_bps: int
    direction: str  # "hike" | "cut"
    instrument: str  # "FFTR_MID" | "MRO" | "DFR"
    excluded: bool
    exclusion_reason: str | None
    audit_note: str | None


def fetch_cbpol_csv(
    ref_area: str,
    *,
    start_period: str = DEFAULT_START_PERIOD,
    client: httpx.Client | None = None,
    retries: int = 3,
    timeout: float = 60.0,
) -> str:
    """Fetch the daily CBPOL history for one jurisdiction as SDMX-CSV text.

    httpx negotiates gzip by default, which this endpoint effectively
    requires: uncompressed full-history responses stall past usable timeouts.
    """
    if ref_area not in REF_AREAS:
        raise ValueError(f"unsupported ref_area {ref_area!r}; expected one of {REF_AREAS}")
    url = CBPOL_URL_TEMPLATE.format(ref_area=ref_area)
    params = {"format": "csv", "startPeriod": start_period}
    owns_client = client is None
    client = client or default_client(timeout)
    try:
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = client.get(url, params=params)
                response.raise_for_status()
                return response.text
            except httpx.HTTPError as error:
                last_error = error
                logger.warning(
                    "CBPOL fetch %s attempt %d/%d failed: %s", ref_area, attempt, retries, error
                )
                if attempt < retries:
                    time.sleep(2**attempt)
        raise RuntimeError(
            f"CBPOL fetch failed for {ref_area} after {retries} attempts"
        ) from last_error
    finally:
        if owns_client:
            client.close()


def parse_cbpol(csv_text: str) -> pd.DataFrame:
    """Parse SDMX-CSV into a clean daily series with columns ``date``, ``level``.

    Uses a real CSV parser (the COMPILATION column contains quoted commas),
    coerces ``NaN`` observation strings to missing, sorts by date, and
    forward-fills so that a shift between consecutive available observations
    is a level change, not a data gap.
    """
    frame = pd.read_csv(io.StringIO(csv_text))
    missing_columns = {"TIME_PERIOD", "OBS_VALUE"} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"CBPOL CSV missing expected columns: {sorted(missing_columns)}")
    series = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["TIME_PERIOD"]).dt.date,
            "level": pd.to_numeric(frame["OBS_VALUE"], errors="coerce"),
        }
    )
    series = series.sort_values("date").reset_index(drop=True)
    n_missing = int(series["level"].isna().sum())
    if n_missing:
        logger.info("forward-filling %d missing observations", n_missing)
    series["level"] = series["level"].ffill()
    return series.dropna(subset=["level"]).reset_index(drop=True)


def xm_instrument(effective: date) -> str:
    """Instrument the XM series tracks on a given date, per its regime table."""
    instrument = XM_REGIMES[0][1]
    for regime_start, name in XM_REGIMES:
        if effective >= regime_start:
            instrument = name
    return instrument


def derive_events(series: pd.DataFrame, ref_area: str) -> list[RateEvent]:
    """Detect level shifts between consecutive available observations."""
    events: list[RateEvent] = []
    previous_level: float | None = None
    for row in series.itertuples(index=False):
        level, effective = float(row.level), row.date
        if previous_level is not None and level != previous_level:
            excluded, reason, audit = False, None, None
            if effective < WINDOW_START[ref_area]:
                excluded = True
                reason = f"outside project window (before {WINDOW_START[ref_area]})"
            elif ref_area == "XM" and effective in XM_REDEFINITION_DATES:
                excluded = True
                reason = "series redefinition (MRO -> DFR), not a policy decision"
            if ref_area == "XM" and effective in XM_AUDIT_DATES:
                audit = (
                    "coincides with an MRO tender-procedure change; "
                    "cross-check against the ECB FM series"
                )
            change_bps = round((level - previous_level) * 100)
            events.append(
                RateEvent(
                    ref_area=ref_area,
                    effective_date=effective,
                    old_level=previous_level,
                    new_level=level,
                    change_bps=change_bps,
                    direction="hike" if change_bps > 0 else "cut",
                    instrument=US_INSTRUMENT if ref_area == "US" else xm_instrument(effective),
                    excluded=excluded,
                    exclusion_reason=reason,
                    audit_note=audit,
                )
            )
            if excluded:
                logger.info("excluded shift %s %s: %s", ref_area, effective, reason)
        previous_level = level
    return events


def build_golden_set(
    out_dir: Path | None = None, *, client: httpx.Client | None = None
) -> pd.DataFrame:
    """Fetch all jurisdictions, derive events, and optionally write the CSV."""
    all_events: list[RateEvent] = []
    for ref_area in REF_AREAS:
        series = parse_cbpol(fetch_cbpol_csv(ref_area, client=client))
        events = derive_events(series, ref_area)
        included = sum(1 for event in events if not event.excluded)
        logger.info(
            "%s: %d shifts detected, %d golden events after exclusions",
            ref_area,
            len(events),
            included,
        )
        all_events.extend(events)
    frame = pd.DataFrame([asdict(event) for event in all_events])
    frame = frame.sort_values(["ref_area", "effective_date"]).reset_index(drop=True)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "cbpol_events.csv"
        frame.to_csv(out_path, index=False)
        logger.info("wrote %s (%d rows)", out_path, len(frame))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CBPOL-derived golden set.")
    parser.add_argument(
        "--out", type=Path, default=Path("data/golden"), help="output directory for the CSV"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    frame = build_golden_set(args.out)
    golden = frame[~frame["excluded"]]
    for ref_area in REF_AREAS:
        subset = golden[golden["ref_area"] == ref_area]
        print(
            f"{ref_area}: {len(subset)} golden events "
            f"({subset['effective_date'].min()} .. {subset['effective_date'].max()})"
        )


if __name__ == "__main__":
    main()
