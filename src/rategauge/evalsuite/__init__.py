"""Evaluation suite: deterministic grading of extraction artifacts against the golden set."""

from rategauge.evalsuite.grader import GoldenSeries, grade_rows
from rategauge.evalsuite.metrics import league_table, mcnemar_exact, summarize

__all__ = ["GoldenSeries", "grade_rows", "league_table", "mcnemar_exact", "summarize"]
