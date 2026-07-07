"""Extraction schema: the structured record models must produce per document.

Two artifacts are kept deliberately separate:

- ``RateDecision`` — the Pydantic model used to validate and type model output.
- ``EXTRACTION_JSON_SCHEMA`` — the hand-written JSON Schema sent to providers.

The JSON Schema is written to satisfy BOTH providers' strict structured-output
constraints (verified 2026-07): every property listed in ``required``,
``additionalProperties: false``, nullability via type arrays, no numeric
min/max or string format keywords. Cross-field consistency (e.g. change_bps
null iff hold) is deliberately NOT hard-enforced here: a model violating it is
a behavior the evaluation measures, not a transport failure.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel

SCHEMA_VERSION = "s1"


class RateDecision(BaseModel):
    """One extracted monetary-policy decision record."""

    bank: Literal["FED", "ECB"]
    decision_date: date | None
    effective_date: date | None
    action: Literal["hike", "cut", "hold", "no_policy_decision"]
    change_bps: int | None
    # FED: the announced federal funds target range, in percent.
    target_range_lower_pct: float | None
    target_range_upper_pct: float | None
    # ECB: all three key rates — the operative instrument changed over time.
    dfr_pct: float | None
    mro_pct: float | None
    mlf_pct: float | None
    evidence_quote: str


EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "bank": {
            "type": "string",
            "enum": ["FED", "ECB"],
            "description": "Issuing central bank of the document.",
        },
        "decision_date": {
            "type": ["string", "null"],
            "description": (
                "Meeting/announcement date as stated in the document text, "
                "formatted YYYY-MM-DD; null if the text does not state it."
            ),
        },
        "effective_date": {
            "type": ["string", "null"],
            "description": (
                "Date the announced rates take effect, as stated in the document, "
                "formatted YYYY-MM-DD; null if not stated."
            ),
        },
        "action": {
            "type": "string",
            "enum": ["hike", "cut", "hold", "no_policy_decision"],
            "description": (
                "hike/cut: the document announces a policy-rate change. "
                "hold: it announces a decision keeping policy rates unchanged. "
                "no_policy_decision: the document does not announce a "
                "monetary-policy rate decision at all."
            ),
        },
        "change_bps": {
            "type": ["integer", "null"],
            "description": (
                "Announced change of the headline policy rate in basis points, "
                "negative for cuts; null for hold and no_policy_decision."
            ),
        },
        "target_range_lower_pct": {
            "type": ["number", "null"],
            "description": (
                "FED only: lower bound of the announced federal funds target "
                "range, in percent. If a single target rate is announced, set "
                "both bounds to that value. null for ECB documents."
            ),
        },
        "target_range_upper_pct": {
            "type": ["number", "null"],
            "description": "FED only: upper bound of the announced target range, in percent.",
        },
        "dfr_pct": {
            "type": ["number", "null"],
            "description": (
                "ECB only: announced deposit facility rate level in percent; "
                "null when not stated."
            ),
        },
        "mro_pct": {
            "type": ["number", "null"],
            "description": (
                "ECB only: announced main refinancing operations rate in percent "
                "(the fixed rate, or the minimum bid rate in the variable-tender "
                "era); null when not stated."
            ),
        },
        "mlf_pct": {
            "type": ["number", "null"],
            "description": (
                "ECB only: announced marginal lending facility rate level in "
                "percent; null when not stated."
            ),
        },
        "evidence_quote": {
            "type": "string",
            "description": (
                "The single verbatim sentence from the document that best "
                "supports the action field."
            ),
        },
    },
    "required": [
        "bank",
        "decision_date",
        "effective_date",
        "action",
        "change_bps",
        "target_range_lower_pct",
        "target_range_upper_pct",
        "dfr_pct",
        "mro_pct",
        "mlf_pct",
        "evidence_quote",
    ],
    "additionalProperties": False,
}
