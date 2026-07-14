"""Provider clients: schema-constrained extraction calls (synchronous path).

Both providers enforce the same hand-written JSON Schema
(``rategauge.schema.EXTRACTION_JSON_SCHEMA``): OpenAI via the Responses API's
Structured Outputs (``text.format`` with ``strict: true``), Anthropic via the
Messages API's ``output_config.format``. TLS is verified against the OS trust
store (corporate networks TLS-intercept; see rategauge.http).
"""

import ssl
import time
from dataclasses import dataclass

import anthropic
import openai
import truststore

from rategauge.schema import EXTRACTION_JSON_SCHEMA

MAX_OUTPUT_TOKENS = 2048


class EmptyResponseError(Exception):
    """The provider returned a response with no usable text payload.

    Carries the response usage: these calls ARE billed, and cost accounting
    (ledger, traces) must not record paid tokens as zero.
    """

    def __init__(self, message: str, *, input_tokens: int = 0, output_tokens: int = 0):
        super().__init__(message)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


@dataclass(frozen=True)
class RawExtraction:
    """One raw model response, before JSON parsing and validation."""

    payload: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


def build_openai_client() -> openai.OpenAI:
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return openai.OpenAI(http_client=openai.DefaultHttpxClient(verify=context))


def build_anthropic_client() -> anthropic.Anthropic:
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return anthropic.Anthropic(http_client=anthropic.DefaultHttpxClient(verify=context))


def extract_openai(
    client: openai.OpenAI, model_id: str, system_prompt: str, document: str
) -> RawExtraction:
    started = time.perf_counter()
    response = client.responses.create(
        model=model_id,
        instructions=system_prompt,
        input=document,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        text={
            "format": {
                "type": "json_schema",
                "name": "rate_decision",
                "schema": EXTRACTION_JSON_SCHEMA,
                "strict": True,
            }
        },
    )
    return RawExtraction(
        payload=response.output_text,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


def extract_anthropic(
    client: anthropic.Anthropic, model_id: str, system_prompt: str, document: str
) -> RawExtraction:
    started = time.perf_counter()
    response = client.messages.create(
        model=model_id,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=system_prompt,
        output_config={"format": {"type": "json_schema", "schema": EXTRACTION_JSON_SCHEMA}},
        messages=[{"role": "user", "content": document}],
    )
    payload = next((block.text for block in response.content if block.type == "text"), None)
    if payload is None:
        # Legitimate SDK shape (e.g. refusal stop reason, max_tokens with no
        # output) — surface as a recordable per-document error, not a crash.
        raise EmptyResponseError(
            f"no text block in response (stop_reason={response.stop_reason})",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
    return RawExtraction(
        payload=payload,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


EXTRACTORS = {"openai": extract_openai, "anthropic": extract_anthropic}
CLIENT_BUILDERS = {"openai": build_openai_client, "anthropic": build_anthropic_client}
