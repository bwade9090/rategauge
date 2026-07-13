"""Grader: join extraction records against the CBPOL-derived golden set.

Implements the matching rules of docs/DESIGN.md section 4:

- A document's expected outcome is the golden decision event whose EFFECTIVE
  date falls inside the announcement window (Fed: 0-5 days forward, ECB: 0-10
  days — shifts land on effective dates, not announcement dates).
- No event in the window means the expected outcome is a HOLD at the
  prevailing level (reconstructed from ALL detected shifts, excluded
  redefinitions included, so levels stay correct across the 2024-09-18
  MRO->DFR switch).
- A document whose window contains an EXCLUDED redefinition shift is
  ungradeable by CBPOL (the real decision is invisible in the series) and is
  flagged, never silently graded.
- US levels are target-range midpoints: extracted ranges are graded as
  (lower + upper) / 2. Euro-area levels are graded against the instrument the
  series tracks on that date (event's instrument, or the regime table for
  holds).
- decision_date is graded against the catalog announcement date (free ground
  truth); a null is an abstention, not an error, and is reported separately.

Every graded row separates three outcomes per numeric field: correct, wrong
(a value was asserted and contradicts the golden set — a hallucination), and
abstained (null). Fabricating a decision on a hold document is flagged as
``fabricated_decision``.
"""

import json
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from rategauge.goldenset.cbpol import xm_instrument
from rategauge.sources.common import read_catalog

GOLDEN_PATH = Path("data/golden/cbpol_events.csv")
CATALOG_PATH = Path("data/catalog/documents.csv")
TRAPS_CATALOG_PATH = Path("data/catalog/traps.csv")

REF_AREA = {"FED": "US", "ECB": "XM"}
WINDOW_DAYS = {"FED": 5, "ECB": 10}
LEVEL_TOLERANCE = 0.005

# Trap documents (official publications that do NOT announce a policy-rate
# decision — FOMC minutes, ECB non-decision monetary-policy releases). The
# only correct extraction is action == "no_policy_decision"; asserting any
# decision (hike, cut, or hold) fabricates a decision record that does not
# exist, which is exactly the false positive the trap set measures.
TRAP_DOC_TYPES = frozenset({"minutes", "non_decision"})

# 2008-12-16: the Fed switched from a point target (1.0%) to a target range
# (0-0.25%). The golden change (-88 bp) is a midpoint-convention artifact no
# document can state, so change_bps is ungradeable for this one event
# (level and action grading remain meaningful).
US_RANGE_TRANSITION_DATES = frozenset({date(2008, 12, 16)})

# Corridor-only ECB decisions: the deposit facility rate moved while the MRO
# — the instrument CBPOL tracked until 2024-09-17 — stayed put, so the
# decision is invisible in the golden series and cannot be graded by it.
# 2015-12-03 (DFR -10bp to -0.30) and 2019-09-12 (DFR -10bp to -0.50) are the
# only such decision releases in the catalog window; upgrading these to
# gradeable via the ECB FM DFR series is a roadmap item.
XM_CORRIDOR_ONLY_ANNOUNCEMENTS = frozenset({date(2015, 12, 3), date(2019, 9, 12)})


@dataclass(frozen=True)
class Shift:
    effective_date: date
    old_level: float
    new_level: float
    change_bps: int
    direction: str
    instrument: str
    excluded: bool


class GoldenSeries:
    """Golden decision events + level reconstruction for one jurisdiction."""

    def __init__(self, shifts: list[Shift]):
        self.shifts = sorted(shifts, key=lambda shift: shift.effective_date)
        self.events = [shift for shift in self.shifts if not shift.excluded]

    @classmethod
    def load_all(cls, path: Path = GOLDEN_PATH) -> dict[str, "GoldenSeries"]:
        frame = pd.read_csv(path, parse_dates=["effective_date"])
        series: dict[str, GoldenSeries] = {}
        for ref_area, group in frame.groupby("ref_area"):
            shifts = [
                Shift(
                    effective_date=row.effective_date.date(),
                    old_level=float(row.old_level),
                    new_level=float(row.new_level),
                    change_bps=int(row.change_bps),
                    direction=str(row.direction),
                    instrument=str(row.instrument),
                    excluded=bool(row.excluded),
                )
                for row in group.itertuples()
            ]
            series[str(ref_area)] = cls(shifts)
        return series

    def in_window(self, start: date, end: date) -> list[Shift]:
        """ALL shifts (excluded ones included) effective within [start, end]."""
        return [s for s in self.shifts if start <= s.effective_date <= end]

    def prevailing_level(self, on_date: date) -> float:
        """Level of the tracked instrument on a date, from the step function."""
        if not self.shifts or on_date < self.shifts[0].effective_date:
            return self.shifts[0].old_level
        level = self.shifts[0].old_level
        for shift in self.shifts:
            if shift.effective_date <= on_date:
                level = shift.new_level
        return level


def load_announcements(catalog_path: Path = CATALOG_PATH) -> dict[str, list[date]]:
    """Sorted announcement dates per bank, for event-ownership resolution.

    Deliberately reads only the decision-document catalog: trap documents
    (minutes are released weeks after the meeting) must never enter the
    ownership timeline, or they would steal golden events from the statements
    that actually announced them.
    """
    announcements: dict[str, list[date]] = {}
    for ref in read_catalog(catalog_path):
        announcements.setdefault(ref.bank, []).append(ref.announcement_date)
    return {bank: sorted(dates) for bank, dates in announcements.items()}


def load_doc_types(
    catalog_path: Path = CATALOG_PATH, traps_path: Path = TRAPS_CATALOG_PATH
) -> dict[str, str]:
    """doc_id -> doc_type across the decision catalog and the trap catalog."""
    doc_types = {ref.doc_id: ref.doc_type for ref in read_catalog(catalog_path)}
    if traps_path.exists():
        doc_types.update({ref.doc_id: ref.doc_type for ref in read_catalog(traps_path)})
    return doc_types


def _owns_event(announced: date, effective: date, bank_announcements: list[date]) -> bool:
    """A document owns an event iff no LATER document was announced by the
    effective date — the event belongs to the announcement closest before it
    (e.g. the 2001-09-18 cut belongs to the 09-17 emergency release, not to
    the 09-13 scheduled hold whose window also covers it)."""
    return bisect_right(bank_announcements, announced) == bisect_right(
        bank_announcements, effective
    )


def grade_rows(
    artifact_rows: list[dict],
    golden: dict[str, GoldenSeries] | None = None,
    announcements: dict[str, list[date]] | None = None,
    doc_types: dict[str, str] | None = None,
) -> list[dict]:
    """Grade one artifact (one model x prompt) row by row."""
    golden = golden or GoldenSeries.load_all()
    announcements = announcements or load_announcements()
    if doc_types is None:
        doc_types = load_doc_types()
        # A doc_id in neither catalog means a stale/missing catalog file
        # (e.g. traps.csv absent): grading it as a control document would
        # silently corrupt every published metric — fail loudly instead.
        unknown = sorted({row["doc_id"] for row in artifact_rows} - doc_types.keys())
        if unknown:
            raise ValueError(
                f"{len(unknown)} artifact doc_id(s) missing from both catalogs "
                f"(is data/catalog/traps.csv present and current?): {unknown[:5]}"
            )
    return [
        _grade_row(
            row,
            golden[REF_AREA[row["bank"]]],
            announcements[row["bank"]],
            is_trap=doc_types.get(row["doc_id"]) in TRAP_DOC_TYPES,
        )
        for row in artifact_rows
    ]


def _grade_row(
    row: dict, series: GoldenSeries, bank_announcements: list[date], *, is_trap: bool = False
) -> dict:
    graded = {
        "doc_id": row["doc_id"],
        "bank": row["bank"],
        "model_key": row["model_key"],
        "prompt_version": row["prompt_version"],
        "announcement_date": row["announcement_date"],
        "status": "graded",
        "expected_kind": None,
        "action_correct": None,
        "fabricated_decision": None,
        "wrong_direction": None,  # events only: asserted hike/cut with wrong sign
        "change_bps": None,  # "correct" | "wrong" | "abstained" (events only)
        "level": None,
        "effective_date": None,
        "decision_date": None,  # "correct" | "wrong" | "abstained" (all docs)
    }
    if is_trap:
        # Stamped before the failure check so trap extraction failures are
        # still attributable to the trap set in the metrics split.
        graded["expected_kind"] = "trap"
    if not row["ok"]:
        graded["status"] = "extraction_failed"
        return graded

    if is_trap:
        # No golden join: the document announces no policy-rate decision, so
        # the only correct action is the abstention label. Rate/date fields
        # are meaningless here (minutes legitimately restate prevailing rates)
        # and stay None.
        action = row["record"].get("action")
        graded["action_correct"] = action == "no_policy_decision"
        graded["fabricated_decision"] = action in ("hike", "cut", "hold")
        return graded

    announced = date.fromisoformat(row["announcement_date"])
    window_end = announced + timedelta(days=WINDOW_DAYS[row["bank"]])
    shifts = series.in_window(announced, window_end)
    events = [
        shift
        for shift in shifts
        if not shift.excluded
        and _owns_event(announced, shift.effective_date, bank_announcements)
    ]
    if any(shift.excluded for shift in shifts) and not events:
        # e.g. the 2024-09-18 MRO->DFR redefinition: the real decision is
        # invisible in the series, so CBPOL cannot grade this document.
        graded["status"] = "ungradeable_redefinition_window"
        return graded
    if row["bank"] == "ECB" and announced in XM_CORRIDOR_ONLY_ANNOUNCEMENTS:
        graded["status"] = "ungradeable_corridor_only"
        return graded

    record = row["record"]
    graded["decision_date"] = _three_way(record.get("decision_date"), announced.isoformat())

    if events:
        event = events[0]  # multi-event ownership does not occur in this corpus
        action = record.get("action")
        graded["expected_kind"] = "change"
        graded["action_correct"] = action == event.direction
        graded["wrong_direction"] = action in ("hike", "cut") and action != event.direction
        if row["bank"] == "FED" and event.effective_date in US_RANGE_TRANSITION_DATES:
            graded["change_bps"] = None  # ungradeable: point-to-range convention switch
        else:
            graded["change_bps"] = _three_way(record.get("change_bps"), event.change_bps)
        graded["level"] = _grade_level(row["bank"], record, event.instrument, event.new_level)
        graded["effective_date"] = _three_way(
            record.get("effective_date"), event.effective_date.isoformat()
        )
    else:
        graded["expected_kind"] = "hold"
        action = record.get("action")
        graded["action_correct"] = action == "hold"
        graded["fabricated_decision"] = action in ("hike", "cut")
        instrument = (
            "FFTR_MID" if row["bank"] == "FED" else xm_instrument(announced)
        )
        graded["level"] = _grade_level(
            row["bank"], record, instrument, series.prevailing_level(announced)
        )
    return graded


def _three_way(got, want) -> str:
    if got is None:
        return "abstained"
    return "correct" if got == want else "wrong"


def _extracted_level(bank: str, record: dict, instrument: str) -> float | None:
    if bank == "FED":
        lower = record.get("target_range_lower_pct")
        upper = record.get("target_range_upper_pct")
        if lower is None or upper is None:
            return None
        return (lower + upper) / 2  # golden US levels are range midpoints
    field = {"DFR": "dfr_pct", "MRO": "mro_pct"}[instrument]
    return record.get(field)


def _grade_level(bank: str, record: dict, instrument: str, want: float) -> str:
    got = _extracted_level(bank, record, instrument)
    if got is None:
        return "abstained"
    return "correct" if abs(got - want) < LEVEL_TOLERANCE else "wrong"


def load_artifact(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
    ]
