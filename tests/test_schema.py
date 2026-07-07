"""Unit tests for the extraction schema (Pydantic model + provider JSON Schema)."""

from datetime import date

import pytest
from pydantic import ValidationError

from rategauge.schema import EXTRACTION_JSON_SCHEMA, RateDecision

VALID_PAYLOAD = {
    "bank": "ECB",
    "decision_date": "2026-06-11",
    "effective_date": "2026-06-17",
    "action": "hike",
    "change_bps": 25,
    "target_range_lower_pct": None,
    "target_range_upper_pct": None,
    "dfr_pct": 2.25,
    "mro_pct": 2.4,
    "mlf_pct": 2.65,
    "evidence_quote": "raise the three key ECB interest rates by 25 basis points",
}


class TestRateDecision:
    def test_valid_payload_validates_and_coerces_dates(self):
        record = RateDecision.model_validate(VALID_PAYLOAD)
        assert record.decision_date == date(2026, 6, 11)
        assert record.effective_date == date(2026, 6, 17)
        assert record.change_bps == 25

    def test_nulls_accepted_for_abstention(self):
        record = RateDecision.model_validate(
            {
                **VALID_PAYLOAD,
                "action": "no_policy_decision",
                "decision_date": None,
                "effective_date": None,
                "change_bps": None,
                "dfr_pct": None,
                "mro_pct": None,
                "mlf_pct": None,
            }
        )
        assert record.action == "no_policy_decision"

    def test_unknown_action_rejected(self):
        with pytest.raises(ValidationError):
            RateDecision.model_validate({**VALID_PAYLOAD, "action": "pivot"})

    def test_missing_field_rejected(self):
        payload = dict(VALID_PAYLOAD)
        del payload["evidence_quote"]
        with pytest.raises(ValidationError):
            RateDecision.model_validate(payload)


class TestProviderJsonSchema:
    """Guard the strict-mode constraints both providers require (DESIGN.md section 5)."""

    def test_all_properties_required(self):
        assert set(EXTRACTION_JSON_SCHEMA["required"]) == set(
            EXTRACTION_JSON_SCHEMA["properties"]
        )

    def test_additional_properties_false(self):
        assert EXTRACTION_JSON_SCHEMA["additionalProperties"] is False

    def test_no_unsupported_keywords(self):
        # Anthropic strict structured outputs reject numeric/string constraints
        # and we avoid format keywords for cross-provider portability.
        forbidden = {"minimum", "maximum", "minLength", "maxLength", "format", "multipleOf"}
        for name, spec in EXTRACTION_JSON_SCHEMA["properties"].items():
            assert not (forbidden & set(spec)), f"{name} uses an unsupported keyword"

    def test_schema_matches_pydantic_fields(self):
        assert set(EXTRACTION_JSON_SCHEMA["properties"]) == set(RateDecision.model_fields)
