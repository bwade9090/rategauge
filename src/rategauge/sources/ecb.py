"""ECB monetary policy decision press releases: enumeration and text extraction.

Enumeration uses the per-year HTML fragments under
``/press/govcdec/mopo/{year}/html/index_include.en.html`` — plain HTTP, no JS,
one ``<dt isoDate>``/``<dd>`` pair per release (the human-facing yearly list
pages are JS shells and unusable). Verified DOM facts (2026-07-07):

- Decision anchors sit inside ``<div class="title">`` with link text exactly
  "Monetary policy decisions"; the same href repeats in the language selector.
- Modern fragments nest "Related" accordions reusing the same dt/dd pattern
  for monetary-policy accounts (``ecb.mg*``) and combined PDFs (``ecb.ds*``);
  the title-text + href rules below reject all of those.
- ``<dt isoDate="...">`` carries the announcement date (BeautifulSoup
  lowercases the attribute to ``isodate``); it matches the date encoded in
  the filename.
- Release pages have exactly one ``<main>``; body text lives in its child
  ``div.section`` / ``div.orderedlist`` blocks (1999-era pages split the text
  across several such blocks — extracting only the first would drop the
  numbered decisions).
"""

import logging
import re
from datetime import date, datetime

import httpx
from bs4 import BeautifulSoup

from rategauge.http import default_client
from rategauge.sources.common import DocumentRef, get_with_retries, normalize_text

logger = logging.getLogger(__name__)

BASE_URL = "https://www.ecb.europa.eu"
FRAGMENT_URL_TEMPLATE = BASE_URL + "/press/govcdec/mopo/{year}/html/index_include.en.html"
FIRST_YEAR = 1999

DECISION_HREF = re.compile(
    r"^/press/pr/date/\d{4}/html/(?:ecb\.mp\d{6}~[0-9a-f]+|pr\d{6}(?:_\d+)?)\.en\.html$"
)
DECISION_TITLE = "Monetary policy decisions"

# Trap-set enumeration (verified 2026-07-08). Non-decision English press
# releases under /press/pr/date/ ALL use an ``ecb.pr`` prefix (with optional
# _N before the tilde) — zero non-decision items exist in the decision URL
# forms (ecb.mp*, legacy pr######). Those forms are deliberately NOT matched
# here, so a decision release whose index title drifts (like the title-case
# 2016-12-08 hold) can never be misclassified into the trap set. Meeting
# accounts (the ECB's minutes) live under /press/accounts/. Combined PDFs
# (ecb.ds*), press-conference statements (ecb.is* — they restate the decisions
# verbatim), and /pub/pdf/ items are rejected by these href rules.
NON_DECISION_HREF = re.compile(
    r"^/press/pr/date/\d{4}/html/ecb\.pr\d{6}(?:_\d+)?~[0-9a-f]+\.en\.html$"
)
ACCOUNTS_HREF = re.compile(r"^/press/accounts/\d{4}/html/ecb\.mg\d{6}~[0-9a-f]+\.en\.html$")

BODY_BLOCK_CLASSES = frozenset({"section", "orderedlist"})


def enumerate_decisions(
    *,
    client: httpx.Client | None = None,
    first_year: int = FIRST_YEAR,
    last_year: int | None = None,
) -> list[DocumentRef]:
    """Enumerate all English 'Monetary policy decisions' releases per year."""
    last_year = last_year or date.today().year
    owns_client = client is None
    client = client or default_client(browser_headers=True)
    refs: dict[str, DocumentRef] = {}
    try:
        for year in range(first_year, last_year + 1):
            url = FRAGMENT_URL_TEMPLATE.format(year=year)
            response = get_with_retries(client, url)
            # Fragments are headerless HTML with no charset declaration.
            soup = BeautifulSoup(response.content, "html.parser", from_encoding="utf-8")
            found = 0
            for anchor in soup.find_all("a"):
                ref = _classify_anchor(anchor)
                if ref is not None and ref.doc_id not in refs:
                    refs[ref.doc_id] = ref
                    found += 1
            logger.info("ECB %d: %d decision releases", year, found)
    finally:
        if owns_client:
            client.close()
    return sorted(refs.values(), key=lambda ref: (ref.announcement_date, ref.doc_id))


def _classify_anchor(anchor) -> DocumentRef | None:
    """Apply the verified keep-rule; return a ref or None.

    Keep iff: parent is div.title AND the normalized title matches the
    decision title case-insensitively AND the href matches the decision
    pattern. This rejects language-selector duplicates, accounts ('Meeting
    of ...'), and combined-statement PDFs. The case-insensitive compare
    matters: the 2016-12-08 hold is titled "Monetary Policy Decisions"
    (title case) — an exact match silently drops it from the corpus.
    """
    if not _in_title_div(anchor) or _normalized_title(anchor) != DECISION_TITLE.lower():
        return None
    href = anchor.get("href") or ""
    if not DECISION_HREF.match(href):
        return None
    return DocumentRef(
        bank="ECB",
        doc_id=_doc_id(href),
        announcement_date=_announcement_date(anchor, href),
        url=BASE_URL + href,
        doc_type="decision",
    )


def enumerate_non_decisions(
    *,
    client: httpx.Client | None = None,
    first_year: int = FIRST_YEAR,
    last_year: int | None = None,
) -> list[DocumentRef]:
    """Trap set: mopo-index items that are NOT rate-decision releases.

    Two doc types from the same yearly fragments: English non-decision press
    releases under /press/pr/date/ (doc_type "non_decision" — programme
    details like PEPP/TPI/TLTRO and remuneration changes; none announce
    key-rate decisions, audited at ingest) and monetary-policy meeting
    accounts (doc_type "minutes"). Items can be filed under a neighboring
    fragment year, so refs are deduped by doc_id and dated by dt isoDate.
    """
    last_year = last_year or date.today().year
    owns_client = client is None
    client = client or default_client(browser_headers=True)
    refs: dict[str, DocumentRef] = {}
    try:
        for year in range(first_year, last_year + 1):
            url = FRAGMENT_URL_TEMPLATE.format(year=year)
            response = get_with_retries(client, url)
            soup = BeautifulSoup(response.content, "html.parser", from_encoding="utf-8")
            found = 0
            for anchor in soup.find_all("a"):
                ref = _classify_trap_anchor(anchor)
                if ref is not None and ref.doc_id not in refs:
                    refs[ref.doc_id] = ref
                    found += 1
            logger.info("ECB %d: %d non-decision trap documents", year, found)
    finally:
        if owns_client:
            client.close()
    return sorted(refs.values(), key=lambda ref: (ref.announcement_date, ref.doc_id))


def _classify_trap_anchor(anchor) -> DocumentRef | None:
    if not _in_title_div(anchor) or _normalized_title(anchor) == DECISION_TITLE.lower():
        return None
    href = anchor.get("href") or ""
    if NON_DECISION_HREF.match(href):
        doc_type = "non_decision"
    elif ACCOUNTS_HREF.match(href):
        doc_type = "minutes"  # meeting accounts are the ECB's minutes
    else:
        return None
    return DocumentRef(
        bank="ECB",
        doc_id=_doc_id(href),
        announcement_date=_announcement_date(anchor, href),
        url=BASE_URL + href,
        doc_type=doc_type,
    )


def _in_title_div(anchor) -> bool:
    parent = anchor.parent
    return parent is not None and parent.name == "div" and "title" in (parent.get("class") or [])


def _normalized_title(anchor) -> str:
    return normalize_text(anchor.get_text(" ", strip=True)).lower()


def _doc_id(href: str) -> str:
    name = href.rsplit("/", 1)[-1]  # ecb.mp260611~4d41bd5e83.en.html | pr050804_2.en.html
    stem = name.split(".en.html")[0].split("~")[0].removeprefix("ecb.")
    return f"ecb_{stem}"


def _announcement_date(anchor, href: str) -> date:
    dd = anchor.find_parent("dd")
    if dd is not None:
        dt = dd.find_previous_sibling("dt")
        iso = dt.get("isodate") if dt is not None else None
        if iso:
            return datetime.strptime(iso, "%Y-%m-%d").date()
    return _date_from_href(href)


def _date_from_href(href: str) -> date:
    digits = re.search(r"(?:ecb\.(?:mp|mg|pr)|pr)(\d{6})", href).group(1)
    yy, month, day = int(digits[:2]), int(digits[2:4]), int(digits[4:6])
    year = 1900 + yy if yy >= 90 else 2000 + yy
    return date(year, month, day)


def extract_text(html: str) -> str:
    """Extract the release body text from a decision page.

    Takes only ``div.section`` / ``div.orderedlist`` blocks inside ``<main>``
    (all eras) and drops the ``***`` separator; headings are kept as context.
    The publication date — the only place these pages state the announcement
    date — is preserved as a leading "Published: ..." line so extraction
    models can ground decision_date instead of guessing.
    """
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")
    if main is None:
        raise ValueError("no <main> element found in ECB page")
    published = _publication_date(main)
    blocks = [
        block
        for block in main.find_all("div", recursive=False)
        if BODY_BLOCK_CLASSES & set(block.get("class") or [])
    ]
    if not blocks:  # defensive: some page wraps blocks one level deeper
        blocks = [
            block
            for block in main.find_all("div")
            if BODY_BLOCK_CLASSES & set(block.get("class") or [])
        ]
    paragraphs: list[str] = []
    if published:
        paragraphs.append(f"Published: {published}")
    for block in blocks:
        for junk in block.find_all("p", class_="ecb-publicationDate"):
            junk.decompose()  # already captured as the "Published:" line
        for element in block.find_all(["p", "li", "h2"]):
            text = normalize_text(element.get_text(" ", strip=True))
            if not text or text == "***":
                continue
            paragraphs.append(text)
    if not paragraphs:
        raise ValueError("no body text extracted from ECB page")
    return "\n\n".join(paragraphs)


def _publication_date(main) -> str | None:
    """Announcement date: p.ecb-publicationDate (modern) or the
    ecb-pressContentPubDate sibling div (retro-converted pre-2015 pages)."""
    for tag, class_name in (("p", "ecb-publicationDate"), ("div", "ecb-pressContentPubDate")):
        node = main.find(tag, class_=class_name)
        if node is not None:
            text = normalize_text(node.get_text(" ", strip=True))
            if text:
                return text
    return None
