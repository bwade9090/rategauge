"""Load documents from catalog + cache into clean text ready for extraction."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rategauge.sources import common, ecb, fed

CATALOG_PATH = Path("data/catalog/documents.csv")
CACHE_DIR = Path("data/cache")

_EXTRACTORS = {"FED": fed.extract_text, "ECB": ecb.extract_text}
_SOURCE_LABELS = {
    "FED": "Federal Reserve (FOMC) press release",
    "ECB": "European Central Bank press release",
}


@dataclass(frozen=True)
class LoadedDocument:
    ref: common.DocumentRef
    text: str

    def as_model_input(self) -> str:
        """The user-message payload: provenance context + document text."""
        return (
            f"Source: {_SOURCE_LABELS[self.ref.bank]}\n\n"
            f"<document>\n{self.text}\n</document>"
        )


def load_documents(
    doc_ids: Sequence[str],
    *,
    catalog_path: Path = CATALOG_PATH,
    cache_dir: Path = CACHE_DIR,
) -> list[LoadedDocument]:
    """Load and text-extract the given documents; loud failure on gaps."""
    catalog = {ref.doc_id: ref for ref in common.read_catalog(catalog_path)}
    cache = common.DocumentCache(cache_dir)
    documents = []
    for doc_id in doc_ids:
        if doc_id not in catalog:
            raise KeyError(f"{doc_id} not in catalog — run `rategauge ingest` first")
        ref = catalog[doc_id]
        html = cache.get(ref)
        if html is None:
            raise FileNotFoundError(f"{doc_id} not cached — run `rategauge ingest` first")
        documents.append(LoadedDocument(ref=ref, text=_EXTRACTORS[ref.bank](html)))
    return documents
