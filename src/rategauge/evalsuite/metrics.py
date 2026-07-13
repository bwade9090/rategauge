"""Metrics: aggregate graded rows into scorecards with statistical rigor.

- Headline rates carry Wilson score 95% confidence intervals. (Percentile
  bootstrap was rejected: it degenerates to a zero-width interval whenever
  the observed rate is exactly 0 or 1 — e.g. 11/11 correct would publish
  "100% [100%, 100%]" where the exact interval is [0.74, 1.0].)
- Model-vs-model differences use McNemar's exact test on paired documents
  (two-sided binomial on the discordant pairs) — the appropriate paired test
  when both models grade the same document set.
"""

from math import comb, sqrt

Z95 = 1.959964


def proportion_ci(outcomes: list[bool]) -> tuple[float, float]:
    """Wilson score 95% interval for a proportion (never degenerate)."""
    n = len(outcomes)
    if n == 0:
        return (float("nan"), float("nan"))
    p = sum(outcomes) / n
    z2 = Z95**2
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z95 * sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(only_a_correct: int, only_b_correct: int) -> float:
    """Two-sided exact McNemar p-value from the discordant-pair counts."""
    n = only_a_correct + only_b_correct
    if n == 0:
        return 1.0
    k = min(only_a_correct, only_b_correct)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)


def _rate(outcomes: list[bool]) -> dict:
    if not outcomes:
        return {"rate": None, "n": 0, "ci95": None}
    lo, hi = proportion_ci(outcomes)
    return {
        "rate": round(sum(outcomes) / len(outcomes), 4),
        "n": len(outcomes),
        "ci95": [round(lo, 4), round(hi, 4)],
    }


def _three_way_rates(values: list[str]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "correct": round(values.count("correct") / n, 4),
        "wrong": round(values.count("wrong") / n, 4),
        "abstained": round(values.count("abstained") / n, 4),
    }


def summarize(graded_rows: list[dict], artifact_rows: list[dict]) -> dict:
    """One scorecard for one (model, prompt) artifact.

    Trap documents (expected_kind == "trap") are reported separately and are
    excluded from every control-document metric, so headline accuracy and
    hallucination rates stay comparable whether or not the trap set was run.
    """
    all_rows = graded_rows
    trap_rows = [row for row in all_rows if row["expected_kind"] == "trap"]
    graded_rows = [row for row in all_rows if row["expected_kind"] != "trap"]
    graded = [row for row in graded_rows if row["status"] == "graded"]
    events = [row for row in graded if row["expected_kind"] == "change"]
    holds = [row for row in graded if row["expected_kind"] == "hold"]

    # Hallucination: a value asserted that contradicts the golden set —
    # a fabricated decision on a hold, a wrong direction on an event, or a
    # wrong change_bps/level. Abstentions (nulls) never count.
    hallucinated = [
        bool(row.get("fabricated_decision"))
        or bool(row.get("wrong_direction"))
        or row.get("change_bps") == "wrong"
        or row.get("level") == "wrong"
        for row in graded
    ]

    return {
        "model_key": all_rows[0]["model_key"] if all_rows else None,
        "prompt_version": all_rows[0]["prompt_version"] if all_rows else None,
        "documents": len(graded_rows),
        "graded": len(graded),
        "extraction_failed": sum(1 for r in graded_rows if r["status"] == "extraction_failed"),
        "ungradeable": sum(1 for r in graded_rows if r["status"].startswith("ungradeable")),
        "events": len(events),
        "holds": len(holds),
        "action_accuracy": _rate([bool(row["action_correct"]) for row in graded]),
        "hallucination_rate": _rate(hallucinated),
        "fabricated_decision_rate": _rate(
            [bool(row["fabricated_decision"]) for row in holds]
        ),
        "change_bps": _three_way_rates(
            [row["change_bps"] for row in events if row["change_bps"] is not None]
        ),
        "level": _three_way_rates([row["level"] for row in graded if row["level"]]),
        "effective_date": _three_way_rates([row["effective_date"] for row in events]),
        "decision_date": _three_way_rates(
            [row["decision_date"] for row in graded if row["decision_date"]]
        ),
        "traps": {
            "documents": len(trap_rows),
            "graded": sum(1 for row in trap_rows if row["status"] == "graded"),
            "extraction_failed": sum(
                1 for row in trap_rows if row["status"] == "extraction_failed"
            ),
            # A fabricated decision on a document that announces none — the
            # false positive the trap set exists to measure. Computed over ALL
            # trap documents (a failed extraction produced no record, hence no
            # fabrication) so every model shares the same denominator.
            "false_positive_rate": _rate(
                [bool(row.get("fabricated_decision")) for row in trap_rows]
            ),
        },
        "cost_usd": round(sum(row["cost_usd"] for row in artifact_rows), 4),
        "input_tokens": sum(row["input_tokens"] for row in artifact_rows),
        "output_tokens": sum(row["output_tokens"] for row in artifact_rows),
    }


def pairwise_mcnemar(graded_by_model: dict[str, list[dict]]) -> list[dict]:
    """McNemar exact tests on action correctness for every model pair."""
    results = []
    keys = sorted(graded_by_model)
    correctness = {
        key: {
            row["doc_id"]: bool(row["action_correct"])
            for row in graded_by_model[key]
            if row["status"] == "graded" and row["expected_kind"] != "trap"
        }
        for key in keys
    }
    for i, model_a in enumerate(keys):
        for model_b in keys[i + 1 :]:
            shared = sorted(set(correctness[model_a]) & set(correctness[model_b]))
            only_a = sum(
                1 for doc in shared if correctness[model_a][doc] and not correctness[model_b][doc]
            )
            only_b = sum(
                1 for doc in shared if correctness[model_b][doc] and not correctness[model_a][doc]
            )
            results.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_paired": len(shared),
                    "only_a_correct": only_a,
                    "only_b_correct": only_b,
                    "p_value": round(mcnemar_exact(only_a, only_b), 4),
                }
            )
    return results


def league_table(summaries: list[dict]) -> str:
    """Markdown league table, best action accuracy first."""
    def fmt_rate(cell: dict) -> str:
        if not cell or cell.get("rate") is None:
            return "-"
        lo, hi = cell["ci95"]
        return f"{cell['rate']:.1%} [{lo:.1%}, {hi:.1%}]"

    def fmt_three(cell: dict, key: str = "correct") -> str:
        if not cell or not cell.get("n"):
            return "-"
        return f"{cell[key]:.1%}"

    lines = [
        "| model | graded | action acc. (95% CI) | hallucination (95% CI) | bps ok "
        "| level ok | eff.date ok | trap FP (95% CI) | cost |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    ranked = sorted(
        summaries, key=lambda s: s["action_accuracy"].get("rate") or 0, reverse=True
    )
    for summary in ranked:
        lines.append(
            f"| {summary['model_key']} "
            f"| {summary['graded']}/{summary['documents']} "
            f"| {fmt_rate(summary['action_accuracy'])} "
            f"| {fmt_rate(summary['hallucination_rate'])} "
            f"| {fmt_three(summary['change_bps'])} "
            f"| {fmt_three(summary['level'])} "
            f"| {fmt_three(summary['effective_date'])} "
            f"| {fmt_rate(summary.get('traps', {}).get('false_positive_rate'))} "
            f"| ${summary['cost_usd']:.2f} |"
        )
    return "\n".join(lines)
