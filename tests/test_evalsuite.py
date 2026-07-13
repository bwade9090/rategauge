"""Unit tests for the grader and metrics (network-free, synthetic golden series)."""

from datetime import date

import pytest

from rategauge.evalsuite import grader, metrics
from rategauge.evalsuite.grader import GoldenSeries, Shift

US_SERIES = GoldenSeries(
    [
        Shift(date(2024, 9, 19), 5.375, 4.875, -50, "cut", "FFTR_MID", False),
        Shift(date(2024, 11, 8), 4.875, 4.625, -25, "cut", "FFTR_MID", False),
    ]
)
XM_SERIES = GoldenSeries(
    [
        Shift(date(2008, 10, 15), 4.25, 3.75, -50, "cut", "MRO", False),
        # The MRO -> DFR series redefinition: excluded, kept for level continuity.
        Shift(date(2024, 9, 18), 4.25, 3.50, -75, "cut", "DFR", True),
        Shift(date(2026, 6, 17), 2.00, 2.25, 25, "hike", "DFR", False),
    ]
)
GOLDEN = {"US": US_SERIES, "XM": XM_SERIES}

# Sorted announcement dates for ownership resolution (hermetic test catalog).
ANNOUNCEMENTS = {
    "FED": [date(2008, 12, 16), date(2024, 9, 18), date(2024, 10, 15),
            date(2024, 10, 16), date(2024, 10, 17)],
    "ECB": [date(2019, 9, 12), date(2024, 9, 12), date(2025, 2, 6), date(2026, 6, 11)],
}


def make_record(**overrides):
    record = {
        "bank": "FED",
        "decision_date": None,
        "effective_date": None,
        "action": "hold",
        "change_bps": None,
        "target_range_lower_pct": None,
        "target_range_upper_pct": None,
        "dfr_pct": None,
        "mro_pct": None,
        "mlf_pct": None,
        "evidence_quote": "quote",
    }
    record.update(overrides)
    return record


def make_row(doc_id, bank, announced, record, ok=True):
    return {
        "doc_id": doc_id,
        "bank": bank,
        "announcement_date": announced,
        "model_key": "test-model",
        "prompt_version": "v001",
        "schema_version": "s1",
        "ok": ok,
        "record": record if ok else None,
        "error": None if ok else "api_error: boom",
        "input_tokens": 100,
        "output_tokens": 10,
        "cost_usd": 0.001,
        "latency_ms": 1,
    }


def grade_one(row):
    # doc_types={} = hermetic: no catalog reads, synthetic ids are controls.
    [graded] = grader.grade_rows([row], GOLDEN, ANNOUNCEMENTS, doc_types={})
    return graded


class TestGoldenSeries:
    def test_prevailing_level_steps_through_all_shifts(self):
        assert XM_SERIES.prevailing_level(date(2008, 1, 1)) == 4.25  # before first shift
        assert XM_SERIES.prevailing_level(date(2010, 1, 1)) == 3.75
        # The excluded redefinition still moves the level — that's the point.
        assert XM_SERIES.prevailing_level(date(2025, 2, 6)) == 3.50
        assert XM_SERIES.prevailing_level(date(2026, 6, 17)) == 2.25

    def test_load_all_from_committed_golden_set(self):
        golden = GoldenSeries.load_all()
        assert len(golden["US"].events) == 74
        assert len(golden["XM"].events) == 59
        assert any(shift.excluded for shift in golden["XM"].shifts)


class TestGradeChangeDocs:
    def test_correct_cut_graded_correct(self):
        record = make_record(
            action="cut", change_bps=-50, decision_date="2024-09-18",
            target_range_lower_pct=4.75, target_range_upper_pct=5.0,
        )
        graded = grade_one(make_row("d1", "FED", "2024-09-18", record))
        assert graded["expected_kind"] == "change"
        assert graded["action_correct"] is True
        assert graded["change_bps"] == "correct"
        assert graded["level"] == "correct"  # midpoint 4.875 vs golden 4.875
        assert graded["effective_date"] == "abstained"
        assert graded["decision_date"] == "correct"

    def test_wrong_bps_is_wrong_not_abstained(self):
        record = make_record(action="cut", change_bps=-25,
                             target_range_lower_pct=4.75, target_range_upper_pct=5.0)
        graded = grade_one(make_row("d1", "FED", "2024-09-18", record))
        assert graded["change_bps"] == "wrong"

    def test_ecb_event_graded_on_regime_instrument(self):
        record = make_record(
            bank="ECB", action="hike", change_bps=25, effective_date="2026-06-17",
            dfr_pct=2.25, mro_pct=2.40, mlf_pct=2.65,
        )
        graded = grade_one(make_row("d1", "ECB", "2026-06-11", record))
        assert graded["action_correct"] is True
        assert graded["level"] == "correct"  # DFR field vs DFR golden level
        assert graded["effective_date"] == "correct"

    def test_wrong_direction_incorrect_action(self):
        record = make_record(action="hike", change_bps=50)
        graded = grade_one(make_row("d1", "FED", "2024-09-18", record))
        assert graded["action_correct"] is False

    def test_point_to_range_transition_bps_ungradeable(self):
        # 2008-12-16: golden -88bp is a midpoint artifact no document can state.
        series = {"US": GoldenSeries(
            [Shift(date(2008, 12, 16), 1.0, 0.125, -88, "cut", "FFTR_MID", False)]
        )}
        record = make_record(action="cut", change_bps=-75,
                             target_range_lower_pct=0.0, target_range_upper_pct=0.25)
        [graded] = grader.grade_rows(
            [make_row("d1", "FED", "2008-12-16", record)], series, ANNOUNCEMENTS, doc_types={}
        )
        assert graded["action_correct"] is True
        assert graded["change_bps"] is None  # excluded from bps stats
        assert graded["level"] == "correct"  # midpoint 0.125 still graded

    def test_wrong_direction_flagged_on_events(self):
        record = make_record(action="hike", change_bps=None)
        graded = grade_one(make_row("d1", "FED", "2024-09-18", record))
        assert graded["wrong_direction"] is True

    def test_event_ownership_goes_to_latest_announcement(self):
        # Real-corpus regression: the 2001-09-17 ECB emergency cut (effective
        # 09-18) also falls inside the 09-13 scheduled hold's 10-day window.
        golden = {"XM": GoldenSeries(
            [Shift(date(2001, 9, 18), 4.25, 3.75, -50, "cut", "MRO", False)]
        )}
        announcements = {"ECB": [date(2001, 9, 13), date(2001, 9, 17)]}
        hold_record = make_record(bank="ECB", action="hold", mro_pct=4.25)
        cut_record = make_record(bank="ECB", action="cut", change_bps=-50, mro_pct=3.75)
        [hold_graded, cut_graded] = grader.grade_rows(
            [
                make_row("ecb_pr010913", "ECB", "2001-09-13", hold_record),
                make_row("ecb_pr010917", "ECB", "2001-09-17", cut_record),
            ],
            golden,
            announcements,
            doc_types={},
        )
        assert hold_graded["expected_kind"] == "hold"  # event owned by the 09-17 doc
        assert hold_graded["action_correct"] is True
        assert hold_graded["level"] == "correct"  # prevailing 4.25 pre-cut
        assert cut_graded["expected_kind"] == "change"
        assert cut_graded["action_correct"] is True


class TestGradeHoldDocs:
    def test_hold_with_prevailing_level_correct(self):
        record = make_record(action="hold",
                             target_range_lower_pct=4.75, target_range_upper_pct=5.0)
        graded = grade_one(make_row("d1", "FED", "2024-10-15", record))
        assert graded["expected_kind"] == "hold"
        assert graded["action_correct"] is True
        assert graded["fabricated_decision"] is False
        assert graded["level"] == "correct"  # prevailing 4.875 after the Sep cut

    def test_fabricated_cut_on_hold_flagged(self):
        record = make_record(action="cut", change_bps=-25)
        graded = grade_one(make_row("d1", "FED", "2024-10-15", record))
        assert graded["action_correct"] is False
        assert graded["fabricated_decision"] is True

    def test_ecb_hold_uses_regime_instrument_after_redefinition(self):
        record = make_record(bank="ECB", action="hold", dfr_pct=3.50)
        graded = grade_one(make_row("d1", "ECB", "2025-02-06", record))
        assert graded["level"] == "correct"  # DFR regime; level crossed the redefinition


class TestGradeTrapDocs:
    def test_trap_abstention_graded_correct(self):
        record = make_record(action="no_policy_decision")
        [graded] = grader.grade_rows(
            [make_row("t1", "FED", "2024-10-15", record)],
            GOLDEN,
            ANNOUNCEMENTS,
            doc_types={"t1": "minutes"},
        )
        assert graded["expected_kind"] == "trap"
        assert graded["action_correct"] is True
        assert graded["fabricated_decision"] is False
        assert graded["level"] is None  # no golden join on trap documents

    @pytest.mark.parametrize("action", ["hike", "cut", "hold"])
    def test_any_asserted_decision_on_trap_is_fabricated(self, action):
        record = make_record(bank="ECB", action=action)
        [graded] = grader.grade_rows(
            [make_row("t1", "ECB", "2025-02-06", record)],
            GOLDEN,
            ANNOUNCEMENTS,
            doc_types={"t1": "non_decision"},
        )
        assert graded["expected_kind"] == "trap"
        assert graded["action_correct"] is False
        assert graded["fabricated_decision"] is True

    def test_trap_extraction_failure_stays_attributed_to_trap_set(self):
        [graded] = grader.grade_rows(
            [make_row("t1", "FED", "2024-10-15", None, ok=False)],
            GOLDEN,
            ANNOUNCEMENTS,
            doc_types={"t1": "minutes"},
        )
        assert graded["status"] == "extraction_failed"
        assert graded["expected_kind"] == "trap"

    def test_decision_doc_types_do_not_trigger_trap_grading(self):
        record = make_record(action="hold")
        [graded] = grader.grade_rows(
            [make_row("h1", "FED", "2024-10-15", record)],
            GOLDEN,
            ANNOUNCEMENTS,
            doc_types={"h1": "statement"},
        )
        assert graded["expected_kind"] == "hold"

    def test_unknown_doc_id_fails_loudly_with_default_catalogs(self):
        # A doc_id in neither catalog (e.g. traps.csv missing on a fresh
        # clone) must never be silently graded as a control document.
        record = make_record(action="hold")
        with pytest.raises(ValueError, match="missing from both catalogs"):
            grader.grade_rows(
                [make_row("zzz_not_in_any_catalog", "FED", "2024-10-15", record)],
                GOLDEN,
                ANNOUNCEMENTS,
            )


class TestUngradeableAndFailures:
    def test_redefinition_window_is_ungradeable(self):
        record = make_record(bank="ECB", action="cut", change_bps=-25)
        graded = grade_one(make_row("d1", "ECB", "2024-09-12", record))
        assert graded["status"] == "ungradeable_redefinition_window"
        assert graded["action_correct"] is None

    def test_corridor_only_decision_is_ungradeable(self):
        # 2019-09-12: DFR-only cut, invisible in the MRO-tracked CBPOL series.
        record = make_record(bank="ECB", action="cut", change_bps=-10, dfr_pct=-0.5)
        graded = grade_one(make_row("d1", "ECB", "2019-09-12", record))
        assert graded["status"] == "ungradeable_corridor_only"
        assert graded["action_correct"] is None

    def test_extraction_failure_passes_through(self):
        graded = grade_one(make_row("d1", "FED", "2024-09-18", None, ok=False))
        assert graded["status"] == "extraction_failed"


class TestMetrics:
    def graded_fixture(self):
        rows = [
            make_row("e1", "FED", "2024-09-18", make_record(
                action="cut", change_bps=-50,
                target_range_lower_pct=4.75, target_range_upper_pct=5.0)),
            make_row("h1", "FED", "2024-10-15", make_record(action="hold")),
            make_row("h2", "FED", "2024-10-16", make_record(action="cut", change_bps=-25)),
            make_row("f1", "FED", "2024-10-17", None, ok=False),
        ]
        return grader.grade_rows(rows, GOLDEN, ANNOUNCEMENTS, doc_types={}), rows

    def test_summarize_counts_and_hallucination(self):
        graded, rows = self.graded_fixture()
        summary = metrics.summarize(graded, rows)
        assert summary["documents"] == 4
        assert summary["graded"] == 3
        assert summary["extraction_failed"] == 1
        assert summary["events"] == 1
        assert summary["holds"] == 2
        assert summary["action_accuracy"]["rate"] == pytest.approx(2 / 3, abs=1e-4)
        # One fabricated decision among three graded docs.
        assert summary["hallucination_rate"]["rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert summary["fabricated_decision_rate"]["rate"] == pytest.approx(0.5)

    def test_wilson_ci_not_degenerate_at_boundary(self):
        lo, hi = metrics.proportion_ci([True] * 11)
        assert hi == 1.0
        assert lo == pytest.approx(0.7412, abs=1e-3)  # never a zero-width interval

    def test_wrong_direction_counts_as_hallucination(self):
        rows = [make_row("e1", "FED", "2024-09-18",
                         make_record(action="hike", change_bps=None))]
        graded = grader.grade_rows(rows, GOLDEN, ANNOUNCEMENTS, doc_types={})
        summary = metrics.summarize(graded, rows)
        assert summary["hallucination_rate"]["rate"] == 1.0

    @pytest.mark.parametrize(
        ("only_a", "only_b", "expected"),
        [(0, 0, 1.0), (1, 1, 1.0), (5, 0, 2 * (1 / 32))],
    )
    def test_mcnemar_exact(self, only_a, only_b, expected):
        assert metrics.mcnemar_exact(only_a, only_b) == pytest.approx(expected)

    def test_league_table_renders_and_ranks(self):
        graded, rows = self.graded_fixture()
        low = metrics.summarize(graded, rows)
        perfect_rows = [rows[0], rows[1]]
        perfect = metrics.summarize(
            grader.grade_rows(perfect_rows, GOLDEN, ANNOUNCEMENTS, doc_types={}), perfect_rows
        )
        perfect["model_key"] = "better-model"
        table = metrics.league_table([low, perfect])
        assert table.index("better-model") < table.index("test-model")
        assert "100.0%" in table

    def test_pairwise_mcnemar_pairs_by_doc_id(self):
        graded, _ = self.graded_fixture()
        better = [dict(row, action_correct=True) for row in graded if row["status"] == "graded"]
        [result] = metrics.pairwise_mcnemar({"a": graded, "b": better})
        assert result["n_paired"] == 3
        assert result["only_b_correct"] == 1
        assert result["only_a_correct"] == 0

    def trap_fixture(self):
        rows = [
            make_row("h1", "FED", "2024-10-15", make_record(action="hold")),
            make_row("t1", "FED", "2024-10-15", make_record(action="no_policy_decision")),
            make_row("t2", "FED", "2024-10-16", make_record(action="hold")),  # false positive
            make_row("t3", "FED", "2024-10-17", None, ok=False),
        ]
        doc_types = {"t1": "minutes", "t2": "minutes", "t3": "non_decision"}
        return grader.grade_rows(rows, GOLDEN, ANNOUNCEMENTS, doc_types=doc_types), rows

    def test_summarize_splits_traps_from_control_metrics(self):
        graded, rows = self.trap_fixture()
        summary = metrics.summarize(graded, rows)
        assert summary["model_key"] == "test-model"
        assert summary["documents"] == 1  # control documents only
        assert summary["graded"] == 1
        assert summary["action_accuracy"]["rate"] == 1.0  # trap mistakes never leak in
        assert summary["hallucination_rate"]["rate"] == 0.0
        assert summary["traps"]["documents"] == 3
        assert summary["traps"]["graded"] == 2
        assert summary["traps"]["extraction_failed"] == 1
        # FP rate over ALL trap documents (same denominator for every model);
        # the failed extraction produced no record, hence no fabrication.
        assert summary["traps"]["false_positive_rate"]["rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert summary["traps"]["false_positive_rate"]["n"] == 3

    def test_summarize_keeps_model_attribution_for_trap_only_artifact(self):
        rows = [make_row("t1", "FED", "2024-10-15", make_record(action="hold"))]
        graded = grader.grade_rows(rows, GOLDEN, ANNOUNCEMENTS, doc_types={"t1": "minutes"})
        summary = metrics.summarize(graded, rows)
        assert summary["model_key"] == "test-model"
        assert summary["prompt_version"] == "v001"
        assert summary["documents"] == 0  # no control documents

    def test_league_table_shows_trap_false_positives(self):
        graded, rows = self.trap_fixture()
        table = metrics.league_table([metrics.summarize(graded, rows)])
        assert "trap FP" in table
        assert "33.3%" in table  # 1 fabrication over all 3 trap documents

    def test_pairwise_mcnemar_excludes_trap_documents(self):
        graded, _ = self.trap_fixture()
        flipped = [
            dict(row, action_correct=not row["action_correct"])
            if row["expected_kind"] == "trap" and row["status"] == "graded"
            else dict(row)
            for row in graded
        ]
        [result] = metrics.pairwise_mcnemar({"a": graded, "b": flipped})
        assert result["n_paired"] == 1  # only the control document pairs
        assert result["p_value"] == 1.0
