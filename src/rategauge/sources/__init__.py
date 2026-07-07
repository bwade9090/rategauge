"""Document sources: enumeration, polite fetching, and text extraction per institution."""

from rategauge.sources.common import (
    DocumentCache,
    DocumentRef,
    fetch_documents,
    normalize_text,
    write_catalog,
)

__all__ = ["DocumentCache", "DocumentRef", "fetch_documents", "normalize_text", "write_catalog"]
