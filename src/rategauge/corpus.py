"""Load documents from catalog + cache into clean text ready for extraction."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rategauge.sources import common, ecb, fed

CATALOG_PATH = Path("data/catalog/documents.csv")
TRAPS_CATALOG_PATH = Path("data/catalog/traps.csv")
CACHE_DIR = Path("data/cache")

_EXTRACTORS = {"FED": fed.extract_text, "ECB": ecb.extract_text}
_SOURCE_LABELS = {
    "FED": "Federal Reserve (FOMC) press release",
    "ECB": "European Central Bank press release",
}
_PUBLICATION_LABELS = {
    "FED": "Federal Reserve (FOMC) publication",
    "ECB": "European Central Bank publication",
}


def _source_label(ref: common.DocumentRef) -> str:
    # Minutes/accounts are not press releases; an honest provenance line
    # matters because it is part of the model input being measured.
    if ref.doc_type == "minutes":
        return _PUBLICATION_LABELS[ref.bank]
    return _SOURCE_LABELS[ref.bank]


def _extract(ref: common.DocumentRef, html: str) -> str:
    if ref.bank == "FED" and ref.doc_type == "minutes":
        return fed.extract_minutes_text(html)  # statements' extractor cannot serve minutes
    return _EXTRACTORS[ref.bank](html)


@dataclass(frozen=True)
class LoadedDocument:
    ref: common.DocumentRef
    text: str

    def as_model_input(self) -> str:
        """The user-message payload: provenance context + document text."""
        return (
            f"Source: {_source_label(self.ref)}\n\n"
            f"<document>\n{self.text}\n</document>"
        )


def load_documents(
    doc_ids: Sequence[str],
    *,
    catalog_path: Path = CATALOG_PATH,
    traps_path: Path = TRAPS_CATALOG_PATH,
    cache_dir: Path = CACHE_DIR,
    fetch_missing: bool = False,
) -> list[LoadedDocument]:
    """Load and text-extract the given documents; loud failure on gaps.

    Looks up doc_ids across the decision catalog and (when present) the trap
    catalog — the two are separate files so trap documents can never leak
    into event-ownership resolution or ``batch submit --all``.
    ``fetch_missing`` fetches cache misses on demand from the official source
    (the serving path: a fresh container starts with an empty cache).
    """
    catalog = {ref.doc_id: ref for ref in common.read_catalog(catalog_path)}
    if traps_path.exists():
        catalog |= {ref.doc_id: ref for ref in common.read_catalog(traps_path)}
    cache = common.DocumentCache(cache_dir)
    documents = []
    for doc_id in doc_ids:
        if doc_id not in catalog:
            raise KeyError(f"{doc_id} not in catalog — run `rategauge ingest` first")
        ref = catalog[doc_id]
        html = cache.get(ref)
        if html is None and fetch_missing:
            html = common.fetch_documents([ref], cache).get(ref.doc_id)
        if html is None:
            raise FileNotFoundError(f"{doc_id} not cached — run `rategauge ingest` first")
        documents.append(LoadedDocument(ref=ref, text=_extract(ref, html)))
    return documents
