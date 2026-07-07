"""Federal Reserve FOMC statements: enumeration and text extraction.

Statement URLs are never synthesized from dates — the pattern changed three
times over 2000-2026 and letter suffixes are not uniform ('b' is the statement
on some emergency dates). Instead, links are scraped from two official
indexes (verified DOM facts, 2026-07-07):

- ``fomccalendars.htm`` (covers ~2021+): each meeting block carries a
  ``<strong>Statement:</strong>`` label with ``PDF | HTML`` links; the HTML
  statement anchor has text exactly "HTML" and an href matching
  ``/newsevents/pressreleases/monetary\\d{8}[a-z].htm``. The single-letter
  suffix requirement rejects Implementation Notes (``...a1.htm``); the exact
  text rule rejects the "Statement on Longer-Run Goals..." notation-vote trap.
- ``fomchistoricalYYYY.htm`` per year (2000-2020): statement anchors have
  text exactly "Statement" (an exact match — prefix matching would catch the
  "Statement on Longer-Run Goals ... (PDF)" trap) with hrefs in three forms:
  boarddocs (``/boarddocs/press/{monetary|general}/YYYY/YYYYMMDD/`` with or
  without a trailing ``default.htm``, used through 2005), old newsevents
  (``/newsevents/press/monetary/YYYYMMDDx.htm``, 2006-2010), and the current
  form. All three verified live against the 2000/2002/2005/2006 year pages.

The announcement date is taken from the URL digits — the date the statement
was released (for conference calls this can be the day after the meeting,
which is the date that matters for grading against effective-dated series).

Content extraction has two parser paths: the modern ``div#article`` template
(serves all 2006+ pages, including old-form URLs) and the legacy boarddocs
table layout (pre-2006), where the statement text sits between two
"DO NOT REMOVE: Wireless Generation" HTML comments inside the only table.
"""

import logging
import re
from collections import Counter
from datetime import date

import httpx
from bs4 import BeautifulSoup

from rategauge.http import default_client
from rategauge.sources.common import DocumentRef, get_with_retries, normalize_text

logger = logging.getLogger(__name__)

BASE_URL = "https://www.federalreserve.gov"
CALENDAR_URL = BASE_URL + "/monetarypolicy/fomccalendars.htm"
HISTORICAL_URL_TEMPLATE = BASE_URL + "/monetarypolicy/fomchistorical{year}.htm"
# The calendar page covers ~2021 onward; earlier years come from per-year
# historical pages (which exist with a multi-year lag — 2020 is the latest).
HISTORICAL_YEARS = tuple(range(2000, 2021))

CALENDAR_HREFS = (
    re.compile(r"^/newsevents/pressreleases/monetary(\d{8})([a-z])\.htm$"),
)
HISTORICAL_HREFS = (
    re.compile(r"^/newsevents/(?:press/monetary/|pressreleases/monetary)(\d{8})([a-z])\.htm$"),
    re.compile(r"^/boarddocs/press/(?:monetary|general)/\d{4}/(\d{8})/(?:default\.htm)?$"),
)

BOARDDOCS_COMMENT = re.compile(r"<!-+[^>]*?DO NOT REMOVE[\s\S]*?-+>")


def enumerate_statements(
    *,
    client: httpx.Client | None = None,
    historical_years: tuple[int, ...] = HISTORICAL_YEARS,
    verify_coverage: bool = True,
) -> list[DocumentRef]:
    """Enumerate FOMC statement URLs from the calendar + historical indexes.

    ``verify_coverage`` guards the calendar/historical handoff: the calendar
    page's coverage floor slides forward each year, and when a year rolls off
    it onto a not-yet-configured historical page, the corpus would silently
    lose ~8 statements. The check makes that gap fail loudly instead.
    """
    owns_client = client is None
    client = client or default_client(browser_headers=True)
    refs: dict[str, DocumentRef] = {}
    try:
        calendar = _get_soup(client, CALENDAR_URL)
        found = _collect(calendar, link_text="HTML", patterns=CALENDAR_HREFS, into=refs)
        logger.info("FED calendar page: %d statements", found)
        for year in historical_years:
            page = _get_soup(client, HISTORICAL_URL_TEMPLATE.format(year=year))
            found = _collect(page, link_text="Statement", patterns=HISTORICAL_HREFS, into=refs)
            logger.info("FED %d: %d statements", year, found)
    finally:
        if owns_client:
            client.close()
    if verify_coverage:
        _verify_yearly_coverage(refs.values())
    return sorted(refs.values(), key=lambda ref: (ref.announcement_date, ref.doc_id))


def _verify_yearly_coverage(refs) -> None:
    """Every completed year since 2000 must have >=4 statements (FOMC floor)."""
    by_year = Counter(ref.announcement_date.year for ref in refs)
    gaps = [year for year in range(2000, date.today().year) if by_year.get(year, 0) < 4]
    if gaps:
        raise RuntimeError(
            f"FOMC statement coverage gap for year(s) {gaps}: the calendar/historical "
            "index handoff has likely moved — extend HISTORICAL_YEARS in sources/fed.py"
        )


def _get_soup(client: httpx.Client, url: str) -> BeautifulSoup:
    return BeautifulSoup(get_with_retries(client, url).text, "html.parser")


def _collect(
    soup: BeautifulSoup, *, link_text: str, patterns: tuple[re.Pattern, ...], into: dict
) -> int:
    found = 0
    for anchor in soup.find_all("a"):
        if anchor.get_text(strip=True) != link_text:
            continue
        href = anchor.get("href") or ""
        match = next((m for m in (p.match(href) for p in patterns) if m is not None), None)
        if match is None:
            continue
        groups = match.groups()
        digits = groups[0]
        suffix = groups[1] if len(groups) > 1 else ""  # boarddocs URLs carry no letter suffix
        doc_id = f"fed_{digits}{suffix}"
        if doc_id in into:
            continue
        into[doc_id] = DocumentRef(
            bank="FED",
            doc_id=doc_id,
            announcement_date=date(int(digits[:4]), int(digits[4:6]), int(digits[6:8])),
            url=BASE_URL + match.group(0),
            doc_type="statement",
        )
        found += 1
    return found


def extract_text(html: str) -> str:
    """Extract the statement text (modern template or legacy boarddocs)."""
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("div", id="article")
    if article is not None:
        return _extract_modern(article)
    return _extract_boarddocs(html, soup)


def _extract_modern(article) -> str:
    """Modern template: the non-heading col-sm-8 div inside #article."""
    body = None
    for candidate in article.find_all("div", class_="col-sm-8"):
        if "heading" not in (candidate.get("class") or []):
            body = candidate
            break
    if body is None:
        raise ValueError("no statement body div found in #article")
    paragraphs: list[str] = []
    # The release date lives in the sibling heading div; keep it so extraction
    # models can ground decision_date instead of guessing.
    time_node = article.find("p", class_="article__time")
    if time_node is not None:
        stamp = normalize_text(time_node.get_text(" ", strip=True))
        if stamp:
            paragraphs.append(f"Published: {stamp}")
    for paragraph in body.find_all("p"):
        # Cloudflare-obfuscated media-inquiries paragraph yields garbage text.
        if paragraph.find("span", class_="__cf_email__") is not None:
            continue
        if paragraph.find("a", href=lambda h: h and h.startswith("/cdn-cgi/")) is not None:
            continue
        text = normalize_text(paragraph.get_text(" ", strip=True))
        if not text:
            continue
        anchors = paragraph.find_all("a")
        if anchors and normalize_text(anchors[0].get_text(" ", strip=True)) == text:
            continue  # link-only paragraph (e.g. "Implementation Note issued ...")
        paragraphs.append(text)
    if not paragraphs:
        raise ValueError("no statement text extracted from modern Fed page")
    return "\n\n".join(paragraphs)


def _extract_boarddocs(html: str, soup: BeautifulSoup) -> str:
    """Legacy pre-2006 layout: slice between the two marker comments."""
    markers = list(BOARDDOCS_COMMENT.finditer(html))
    if len(markers) >= 2:
        fragment = BeautifulSoup(html[markers[0].end() : markers[1].start()], "html.parser")
        paragraphs = []
        release_line = soup.find(string=re.compile(r"Release Date:"))
        if release_line:
            paragraphs.append(normalize_text(str(release_line)))
        for paragraph in fragment.find_all("p"):
            # Legacy pages leave <p> unclosed, so html.parser NESTS each
            # paragraph inside the previous one — get_text() on each would
            # re-emit every following paragraph. Take only own text.
            own = " ".join(
                str(string)
                for string in paragraph.find_all(string=True)
                if string.find_parent("p") is paragraph
            )
            text = normalize_text(own)
            if text:
                paragraphs.append(text)
        if paragraphs:
            return "\n\n".join(paragraphs)
    # Fallback: the only table's first cell, trimmed at the footer.
    table = soup.find("table")
    if table is None:
        raise ValueError("unrecognized Fed page layout")
    text = normalize_text(table.find("td").get_text(" ", strip=True))
    text = re.split(r"Last update:", text)[0]
    if not text:
        raise ValueError("no statement text extracted from boarddocs Fed page")
    return text
