"""Shared plumbing for document sources: references, local cache, polite fetching.

Documents are fetched on demand from official sources and cached locally under
``data/cache/`` (gitignored — nothing is re-hosted). The enumerated catalog
(bank, date, URL — facts only) is committed for reproducibility.
"""

import csv
import logging
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

from rategauge.http import default_client

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Collapse whitespace and neutralize typographic landmines.

    Observed in the wild: NBSP inside 2008-era Fed sentences, U+2011
    non-breaking hyphens in 2025 statements ("mortgage-backed").
    """
    text = text.replace("\xa0", " ").replace("‑", "-")
    return re.sub(r"\s+", " ", text).strip()

POLITE_DELAY_SECONDS = 0.5

CATALOG_COLUMNS = ["bank", "doc_id", "announcement_date", "url", "doc_type"]


@dataclass(frozen=True)
class DocumentRef:
    """One official document, identified before any fetching happens."""

    bank: str  # "FED" | "ECB"
    doc_id: str  # stable slug, e.g. "fed_20250618a", "ecb_mp260611"
    announcement_date: date
    url: str
    doc_type: str  # "statement" | "decision"


def get_with_retries(client: httpx.Client, url: str, *, retries: int = 3) -> httpx.Response:
    """GET with retry/backoff; raises RuntimeError after exhausting retries."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.get(url)
            response.raise_for_status()
            return response
        except httpx.HTTPError as error:
            last_error = error
            logger.warning("GET %s attempt %d/%d failed: %s", url, attempt, retries, error)
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"GET failed after {retries} attempts: {url}") from last_error


class DocumentCache:
    """Filesystem cache: one HTML file per document, keyed by doc_id."""

    def __init__(self, root: Path):
        self.root = root

    def path_for(self, ref: DocumentRef) -> Path:
        return self.root / ref.bank.lower() / f"{ref.doc_id}.html"

    def get(self, ref: DocumentRef) -> str | None:
        path = self.path_for(ref)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def put(self, ref: DocumentRef, html: str) -> Path:
        path = self.path_for(ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return path


def fetch_documents(
    refs: Sequence[DocumentRef],
    cache: DocumentCache,
    *,
    client: httpx.Client | None = None,
    delay_seconds: float = POLITE_DELAY_SECONDS,
    retries: int = 3,
) -> dict[str, str]:
    """Return doc_id -> HTML for every ref, fetching only what the cache lacks.

    Network fetches are spaced by ``delay_seconds``; failures after retries are
    logged and skipped (the caller sees which doc_ids are missing), so one dead
    URL cannot kill a 500-document backfill.
    """
    owns_client = client is None
    client = client or default_client(browser_headers=True)
    results: dict[str, str] = {}
    fetched = skipped = 0
    try:
        for ref in refs:
            cached = cache.get(ref)
            if cached is not None:
                results[ref.doc_id] = cached
                continue
            html = _fetch_one(ref, client, retries)
            if html is None:
                skipped += 1
                continue
            cache.put(ref, html)
            results[ref.doc_id] = html
            fetched += 1
            if fetched % 25 == 0:
                logger.info("fetched %d documents (last: %s)", fetched, ref.doc_id)
            time.sleep(delay_seconds)
    finally:
        if owns_client:
            client.close()
    logger.info(
        "documents ready: %d total (%d newly fetched, %d failed)",
        len(results),
        fetched,
        skipped,
    )
    return results


def _fetch_one(ref: DocumentRef, client: httpx.Client, retries: int) -> str | None:
    try:
        return get_with_retries(client, ref.url, retries=retries).text
    except RuntimeError:
        logger.error("giving up on %s (%s)", ref.doc_id, ref.url)
        return None


def read_catalog(path: Path) -> list[DocumentRef]:
    """Read a previously written catalog back into DocumentRefs."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            DocumentRef(
                bank=row["bank"],
                doc_id=row["doc_id"],
                announcement_date=date.fromisoformat(row["announcement_date"]),
                url=row["url"],
                doc_type=row["doc_type"],
            )
            for row in csv.DictReader(handle)
        ]


def write_catalog(refs: Iterable[DocumentRef], out_path: Path) -> None:
    """Write the enumerated document catalog (facts only: dates and URLs)."""
    rows = sorted(refs, key=lambda r: (r.bank, r.announcement_date, r.doc_id))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CATALOG_COLUMNS)
        for ref in rows:
            writer.writerow(
                [ref.bank, ref.doc_id, ref.announcement_date.isoformat(), ref.url, ref.doc_type]
            )
    logger.info("wrote catalog %s (%d documents)", out_path, len(rows))
