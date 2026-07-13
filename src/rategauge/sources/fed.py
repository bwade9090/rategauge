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
from datetime import date, datetime

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

# Minutes links (trap set). Verified DOM facts (2026-07-08): the calendar page
# links minutes as text "HTML" inside div.fomc-meeting__minutes; historical
# pages link text "Minutes" -> /fomc/minutes/YYYYMMDD.htm for meetings through
# 2007-09-18 and text "HTML" -> /monetarypolicy/fomcminutesYYYYMMDD.htm from
# 2007-10-31 on. The strict href regexes reject the observed traps: Beige Book
# links (text also "HTML", /monetarypolicy/beigebook/...), SEP projections
# (/monetarypolicy/fomcprojtabl...), PDF variants, and conference-call minutes
# folded into another document as absolute-URL/#fragment links.
MINUTES_HREFS = (
    re.compile(r"^/monetarypolicy/fomcminutes(\d{8})\.htm$"),
    re.compile(r"^/fomc/minutes/(\d{8})\.htm$"),
)
# URL digits encode the MEETING date; the release lags by 3 weeks (modern) to
# 7 weeks (2000-2004). The release date lives in the index "(Released ...)"
# text next to each minutes link — full month names 2000-2004 and 2011+,
# abbreviated 2005-2010.
MINUTES_RELEASED_TEXT = re.compile(r"\(Released ([^)]+)\)")
MINUTES_SAMPLE_YEARS = tuple(range(2000, 2025))


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


def enumerate_minutes(
    *,
    client: httpx.Client | None = None,
    historical_years: tuple[int, ...] = HISTORICAL_YEARS,
    sample_years: tuple[int, ...] = MINUTES_SAMPLE_YEARS,
) -> list[DocumentRef]:
    """FOMC minutes of the first scheduled meeting of each sample year (traps).

    Minutes never announce a decision — they recount one, weeks later — so the
    only correct extraction is action == "no_policy_decision". The sampling
    rule (earliest meeting per year) is deterministic and documented here;
    announcement_date is the RELEASE date parsed from the index "(Released
    ...)" text, because the URL digits encode the meeting date instead.
    """
    owns_client = client is None
    client = client or default_client(browser_headers=True)
    refs: dict[str, DocumentRef] = {}
    try:
        calendar = _get_soup(client, CALENDAR_URL)
        _collect_minutes(calendar, link_text="HTML", into=refs)
        for year in historical_years:
            page = _get_soup(client, HISTORICAL_URL_TEMPLATE.format(year=year))
            _collect_minutes(page, link_text="Minutes", into=refs)
            _collect_minutes(page, link_text="HTML", into=refs)
    finally:
        if owns_client:
            client.close()
    return sorted(
        _first_meeting_per_year(refs.values(), sample_years), key=lambda ref: ref.doc_id
    )


def _collect_minutes(soup: BeautifulSoup, *, link_text: str, into: dict) -> None:
    for anchor in soup.find_all("a"):
        if anchor.get_text(strip=True) != link_text:
            continue
        href = anchor.get("href") or ""
        match = next((m for m in (p.match(href) for p in MINUTES_HREFS) if m is not None), None)
        if match is None:
            continue
        digits = match.group(1)
        doc_id = f"fed_min_{digits}"  # namespaced: boarddocs statements already own fed_{digits}
        if doc_id in into:
            continue
        meeting = date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
        into[doc_id] = DocumentRef(
            bank="FED",
            doc_id=doc_id,
            announcement_date=_minutes_release_date(anchor, meeting),
            url=BASE_URL + href,
            doc_type="minutes",
        )


def _minutes_release_date(anchor, meeting: date) -> date:
    node = anchor.parent
    for _ in range(4):  # the "(Released ...)" text sits in a nearby ancestor block
        if node is None:
            break
        match = MINUTES_RELEASED_TEXT.search(node.get_text(" ", strip=True))
        if match:
            parsed = _parse_release_date(match.group(1))
            if parsed is not None:
                return parsed
        node = node.parent
    logger.warning("no release date found for minutes of %s; falling back to meeting date",
                   meeting)
    return meeting


def _parse_release_date(text: str) -> date | None:
    text = normalize_text(text).replace(".", "")
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _meeting_date(doc_id: str) -> date:
    digits = doc_id.removeprefix("fed_min_")
    return date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))


def _first_meeting_per_year(refs, sample_years: tuple[int, ...]) -> list[DocumentRef]:
    by_year: dict[int, DocumentRef] = {}
    for ref in refs:
        meeting = _meeting_date(ref.doc_id)
        if meeting.year not in sample_years:
            continue
        current = by_year.get(meeting.year)
        if current is None or meeting < _meeting_date(current.doc_id):
            by_year[meeting.year] = ref
    missing = [year for year in sample_years if year not in by_year]
    if missing:
        raise RuntimeError(f"no FOMC minutes found for year(s) {missing} — index layout moved?")
    late = [ref.doc_id for ref in by_year.values() if _meeting_date(ref.doc_id).month > 3]
    if late:
        # First scheduled meetings are always Jan/Feb; a later pick means the
        # index dropped entries or an unscheduled meeting leaked in.
        raise RuntimeError(f"first-meeting sample picked a post-March meeting: {late}")
    return list(by_year.values())


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


def extract_minutes_text(html: str) -> str:
    """Extract FOMC minutes text across all three page templates (2026-07-08).

    Modern pages put the body as direct <p>/<blockquote> children of
    div#article (no col-sm-8 wrapper, so ``extract_text`` cannot serve them);
    2008-2010-vintage pages use div#leftText; legacy /fomc/minutes/ pages have
    no container ids at all, and pre-2003 files leave most <p> unclosed
    (html.parser nests them), so text is taken per element via the own-text
    rule everywhere.
    """
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("div", id="article") or soup.find("div", id="leftText") or soup
    paragraphs: list[str] = []
    published = _minutes_release_line(soup)
    if published:
        paragraphs.append(published)
    for element in container.find_all(["p", "li", "blockquote"]):
        own = " ".join(
            str(string)
            for string in element.find_all(string=True)
            if string.find_parent(["p", "li", "blockquote"]) is element
        )
        text = normalize_text(own)
        if not text or text == "Return to top":
            continue
        if text.startswith(("Last update:", "Last Update:")):
            continue  # footer; already captured as the "Published:" line
        anchors = element.find_all("a")
        if anchors and normalize_text(anchors[0].get_text(" ", strip=True)) == text:
            continue  # link-only paragraph (accessible-materials tail etc.)
        paragraphs.append(text)
    body = "\n\n".join(paragraphs)
    if len(body) < 5000:
        # Real minutes run tens of thousands of characters; a short result
        # means a template drifted and we silently grabbed navigation junk.
        raise ValueError(f"minutes extraction suspiciously short ({len(body)} chars)")
    return body


def _minutes_release_line(soup: BeautifulSoup) -> str | None:
    """The release date: div#lastUpdate (2008+) or the footer 'Last update:'
    string (legacy) — verified equal to the index-stated release date."""
    node = soup.find("div", id="lastUpdate")
    text = normalize_text(node.get_text(" ", strip=True)) if node is not None else None
    if not text:
        line = soup.find(string=re.compile(r"Last [Uu]pdate:"))
        text = normalize_text(str(line)) if line else None
    if not text:
        return None
    return "Published: " + re.sub(r"^Last [Uu]pdate:\s*", "", text)


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
