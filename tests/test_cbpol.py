"""Unit tests for the CBPOL golden-set derivation (network-free).

Fixtures replicate the exact SDMX-CSV shape observed on the live BIS API on
2026-07-06, including the quoted-comma COMPILATION column, weekend NaN
observations, and the missing-day / redefinition quirks documented in
docs/DESIGN.md section 3.1.
"""

import os
from datetime import date

import httpx
import pandas as pd
import pytest

from rategauge.goldenset import cbpol

EXPECTED_COLUMNS = [
    "ref_area",
    "effective_date",
    "old_level",
    "new_level",
    "change_bps",
    "direction",
    "instrument",
    "excluded",
    "exclusion_reason",
    "audit_note",
]

HEADER = (
    "FREQ,REF_AREA,UNIT_MEASURE,UNIT_MULT,TIME_FORMAT,COMPILATION,DECIMALS,"
    "SOURCE_REF,SUPP_INFO_BREAKS,TITLE,TIME_PERIOD,OBS_VALUE,OBS_STATUS,OBS_CONF,OBS_PRE_BREAK"
)


def make_csv(ref_area: str, observations: list[tuple[str, str]]) -> str:
    """Build an SDMX-CSV payload like the live API's, quoted commas included."""
    compilation = '"From 19 Dec 1985 onwards: mid-point, of the target rate."'
    rows = [
        f"D,{ref_area},368,0,,{compilation},4,Source,,Title,{period},{value},A,F,"
        for period, value in observations
    ]
    return "\n".join([HEADER, *rows])


class TestParse:
    def test_quoted_commas_and_types(self):
        series = cbpol.parse_cbpol(make_csv("US", [("2024-09-18", "5.375")]))
        assert list(series.columns) == ["date", "level"]
        assert series.loc[0, "date"] == date(2024, 9, 18)
        assert series.loc[0, "level"] == 5.375

    def test_sorts_by_date(self):
        series = cbpol.parse_cbpol(
            make_csv("US", [("2024-09-19", "4.875"), ("2024-09-18", "5.375")])
        )
        assert series["date"].tolist() == [date(2024, 9, 18), date(2024, 9, 19)]

    def test_nan_observations_forward_filled(self):
        # Replicates the 32 XM weekend NaNs seen between 2024-09-21 and 2025-01-05.
        series = cbpol.parse_cbpol(
            make_csv(
                "XM",
                [
                    ("2024-09-20", "3.50"),
                    ("2024-09-21", "NaN"),
                    ("2024-09-22", "NaN"),
                    ("2024-09-23", "3.50"),
                ],
            )
        )
        assert series["level"].tolist() == [3.50, 3.50, 3.50, 3.50]

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError, match="missing expected columns"):
            cbpol.parse_cbpol("A,B\n1,2")


class TestDeriveEvents:
    def test_us_cut_detected_on_effective_date(self):
        # FOMC announced 2024-09-18; the midpoint shifts on the 19th (effective date).
        series = cbpol.parse_cbpol(
            make_csv(
                "US",
                [
                    ("2024-09-17", "5.375"),
                    ("2024-09-18", "5.375"),
                    ("2024-09-19", "4.875"),
                    ("2024-09-20", "4.875"),
                ],
            )
        )
        events = cbpol.derive_events(series, "US")
        assert len(events) == 1
        event = events[0]
        assert event.effective_date == date(2024, 9, 19)
        assert event.change_bps == -50
        assert event.direction == "cut"
        assert event.instrument == "FFTR_MID"
        assert not event.excluded

    def test_hold_produces_no_events(self):
        series = cbpol.parse_cbpol(
            make_csv("US", [("2025-01-01", "4.375"), ("2025-01-02", "4.375")])
        )
        assert cbpol.derive_events(series, "US") == []

    def test_nan_gap_produces_no_spurious_events(self):
        series = cbpol.parse_cbpol(
            make_csv(
                "XM",
                [
                    ("2024-11-01", "3.25"),
                    ("2024-11-02", "NaN"),
                    ("2024-11-03", "NaN"),
                    ("2024-11-04", "3.25"),
                ],
            )
        )
        assert cbpol.derive_events(series, "XM") == []

    def test_change_across_missing_day_dated_next_available(self):
        # Replicates the single missing US day (2024-10-18): a change across a
        # hole must be dated on the next available observation.
        series = cbpol.parse_cbpol(
            make_csv("US", [("2024-10-17", "4.875"), ("2024-10-19", "4.625")])
        )
        events = cbpol.derive_events(series, "US")
        assert len(events) == 1
        assert events[0].effective_date == date(2024, 10, 19)
        assert events[0].change_bps == -25

    def test_pre_window_event_excluded_with_reason(self):
        series = cbpol.parse_cbpol(
            make_csv("US", [("1999-11-15", "5.25"), ("1999-11-16", "5.50")])
        )
        events = cbpol.derive_events(series, "US")
        assert len(events) == 1
        assert events[0].excluded
        assert "outside project window" in events[0].exclusion_reason

    def test_xm_redefinition_excluded_but_kept(self):
        # 2024-09-18: series switches MRO -> DFR; apparent -75bp is an artifact.
        series = cbpol.parse_cbpol(
            make_csv("XM", [("2024-09-17", "4.25"), ("2024-09-18", "3.50")])
        )
        events = cbpol.derive_events(series, "XM")
        assert len(events) == 1
        event = events[0]
        assert event.excluded
        assert "redefinition" in event.exclusion_reason
        assert event.change_bps == -75  # recorded faithfully, excluded transparently

    def test_xm_audit_date_kept_with_note(self):
        # 2008-10-15: tender-procedure boundary AND a genuine -50bp cut — kept.
        series = cbpol.parse_cbpol(
            make_csv("XM", [("2008-10-14", "4.25"), ("2008-10-15", "3.75")])
        )
        events = cbpol.derive_events(series, "XM")
        assert len(events) == 1
        assert not events[0].excluded
        assert "cross-check" in events[0].audit_note

    def test_fractional_bps_rounding(self):
        series = cbpol.parse_cbpol(
            make_csv("US", [("2025-06-01", "4.375"), ("2025-06-02", "4.125")])
        )
        assert cbpol.derive_events(series, "US")[0].change_bps == -25


class TestXmInstrument:
    @pytest.mark.parametrize(
        ("effective", "expected"),
        [
            (date(1999, 6, 1), "MRO"),
            (date(2005, 3, 1), "MRO"),
            (date(2024, 9, 17), "MRO"),
            (date(2024, 9, 18), "DFR"),
            (date(2026, 6, 17), "DFR"),
        ],
    )
    def test_regime_mapping(self, effective, expected):
        assert cbpol.xm_instrument(effective) == expected


class TestBuildGoldenSet:
    """Offline end-to-end test of build_golden_set via httpx.MockTransport."""

    US_CSV = make_csv(
        "US",
        [
            ("1999-11-15", "5.25"),
            ("1999-11-16", "5.50"),  # pre-window hike -> excluded, kept
            ("2024-09-18", "5.50"),
            ("2024-09-19", "5.00"),  # -50bp cut -> golden
        ],
    )
    XM_CSV = make_csv(
        "XM",
        [
            ("2024-09-17", "4.25"),
            ("2024-09-18", "3.50"),  # MRO -> DFR redefinition -> excluded, kept
            ("2026-06-16", "3.50"),
            ("2026-06-17", "3.75"),  # +25bp hike -> golden
        ],
    )

    def build(self, tmp_path):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params["format"] == "csv"
            payload = self.US_CSV if "D.US" in request.url.path else self.XM_CSV
            return httpx.Response(200, text=payload)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        return cbpol.build_golden_set(tmp_path, client=client)

    def test_exclusions_kept_never_silently_dropped(self, tmp_path):
        frame = self.build(tmp_path)
        assert len(frame) == 4
        excluded = frame[frame["excluded"]]
        assert len(excluded) == 2
        assert excluded["exclusion_reason"].notna().all()
        golden = frame[~frame["excluded"]]
        assert {(row.ref_area, row.change_bps) for row in golden.itertuples()} == {
            ("US", -50),
            ("XM", 25),
        }

    def test_written_csv_round_trips_with_documented_shape(self, tmp_path):
        frame = self.build(tmp_path)
        assert list(frame.columns) == EXPECTED_COLUMNS
        on_disk = pd.read_csv(tmp_path / "cbpol_events.csv")
        assert list(on_disk.columns) == EXPECTED_COLUMNS
        assert len(on_disk) == 4
        assert int(on_disk["excluded"].sum()) == 2
        # sorted by ref_area then effective_date, as documented
        assert on_disk["ref_area"].tolist() == sorted(on_disk["ref_area"].tolist())


@pytest.mark.skipif(
    os.getenv("RATEGAUGE_LIVE", "").lower() not in {"1", "true", "yes"},
    reason="live BIS API test; set RATEGAUGE_LIVE=1 to run",
)
class TestLive:
    """Sanity checks against the live BIS API (docs/DESIGN.md section 4)."""

    def test_golden_set_matches_known_history(self, tmp_path):
        frame = cbpol.build_golden_set(tmp_path)
        golden = frame[~frame["excluded"]]
        us = golden[golden["ref_area"] == "US"]
        xm = golden[golden["ref_area"] == "XM"]

        # Snapshot 2026-07-06: 74 US events since 2000; ~59 XM after exclusions.
        assert 74 <= len(us) <= 80
        assert 55 <= len(xm) <= 65

        # Known recent events, triple-verified across BIS/ECB/Fed sources.
        us_events = {(str(row.effective_date), row.change_bps) for row in us.itertuples()}
        xm_events = {(str(row.effective_date), row.change_bps) for row in xm.itertuples()}
        assert ("2025-12-11", -25) in us_events  # FOMC 2025-12-10 cut, effective +1d
        assert ("2026-06-17", 25) in xm_events  # ECB 2026-06-11 hike, effective 06-17

        # The redefinition artifact must be excluded, never golden.
        assert "2024-09-18" not in {str(d) for d in xm["effective_date"]}
