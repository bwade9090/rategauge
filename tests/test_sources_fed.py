"""Unit tests for Fed enumeration and extraction (network-free).

Fixtures replicate verbatim DOM quirks observed live on 2026-07-07: the
malformed stray quote in fomc-meeting divs, the "Statement on Longer-Run
Goals" trap links (calendar notation-vote row and historical PDF), minutes
links whose text is also "HTML", Implementation Note ``a1.htm`` links,
boarddocs URL variants (monetary/general, with/without default.htm), and the
Cloudflare-obfuscated media-inquiries paragraph.
"""

import os
from datetime import date

import httpx
import pytest

from rategauge.sources import fed

CALENDAR_PAGE = """<html><body>
<div class="panel panel-default"><div class="panel-heading"><h4><a id="42827">2025 FOMC Meetings</a></h4></div>
<div class="row fomc-meeting" ">
<div class="fomc-meeting__month col-xs-5"><strong>June</strong></div>
<div class="fomc-meeting__date col-xs-4">17-18*</div>
<div class="col-xs-12 col-md-4 col-lg-2">
<strong>Statement:</strong><br>
<a href="/monetarypolicy/files/monetary20250618a1.pdf">PDF</a> | <a href="/newsevents/pressreleases/monetary20250618a.htm">HTML</a><br>
<a href="/newsevents/pressreleases/monetary20250618a1.htm">Implementation Note</a>
</div>
<div class="col-xs-12 col-md-4 col-lg-4 fomc-meeting__minutes">Minutes: <a href="/monetarypolicy/fomcminutes20250618.htm">HTML</a> (Released July 9, 2025)</div>
</div>
<div class="row fomc-meeting" ">
<div class="fomc-meeting__date col-xs-4">22 (notation vote)</div>
<a href="/newsevents/pressreleases/monetary20250822a.htm">Statement on Longer-Run Goals and Monetary Policy Strategy</a>
</div>
</div></body></html>"""

HISTORICAL_PAGE = """<html><body>
<div class="panel panel-default"><div class="panel-heading"><h5>January 21 Conference Call - 2008</h5></div>
<div class="row divided-row panel-body">
<p><a href="/newsevents/press/monetary/20080122b.htm">Statement</a></p>
<p><a href="/monetarypolicy/files/FOMC_LongerRunGoals_201501.pdf">Statement on Longer-Run Goals and Monetary Policy Strategy (PDF)</a></p>
<p><a href="/boarddocs/press/general/2000/20000516/">Statement</a></p>
<p><a href="/boarddocs/press/monetary/2005/20050202/default.htm">Statement</a></p>
<p>Minutes: <a href="/monetarypolicy/fomcminutes20080130.htm">HTML</a></p>
</div></div></body></html>"""

MODERN_PAGE = """<html><body><div id="article">
<div class="heading col-xs-12 col-sm-8 col-md-8"><p class="article__time">June 18, 2025</p>
<h3 class="title">Federal Reserve issues FOMC statement</h3></div>
<div class="col-xs-12 col-sm-4 col-md-4 hidden-sm"></div>
<div class="col-xs-12 col-sm-8 col-md-8">
<P></P>
<p>Recent indicators suggest that economic activity has continued to expand at a solid pace.</p>
<p>The Committee decided to maintain the target range for the federal funds rate at 4-1/4 to 4 1/2 percent.</p>
<p>For media inquiries, please email <a href="/cdn-cgi/l/email-protection#ab"><span class="__cf_email__" data-cfemail="ab">[email protected]</span></a> or call 202-452-2955.</p>
<p><a href="/newsevents/pressreleases/monetary20250618a1.htm">Implementation Note issued June 18, 2025</a></p>
</div></div></body></html>"""

BOARDDOCS_PAGE = """<HTML><BODY BGCOLOR="#FFFFFF">
<FONT SIZE=+1><I>Release Date: November 6, 2002</I></FONT>
<TABLE WIDTH="600"><TR><TD>
<!------------DO NOT REMOVE:  Wireless Generation------------->
<p>The Federal Open Market Committee decided today to lower its target for the federal funds rate by 50 basis points to 1 1/4 percent.
<p>In a related action, the Board of Governors approved a 50 basis point reduction in the discount rate.
<!------------DO NOT REMOVE:  Wireless Generation------------->
<a href="../default.htm">2002 Monetary policy</a><HR NOSHADE>
<font size="-1">Home | News and events<br><b>Last update: November 6, 2002</b></font>
</TD></TR></TABLE></BODY></HTML>"""


class TestEnumerate:
    def enumerate(self):
        def handler(request: httpx.Request) -> httpx.Response:
            page = CALENDAR_PAGE if "fomccalendars" in str(request.url) else HISTORICAL_PAGE
            return httpx.Response(200, text=page)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        return fed.enumerate_statements(
            client=client, historical_years=(2008,), verify_coverage=False
        )

    def test_coverage_gap_fails_loudly(self):
        # When the calendar/historical handoff moves, enumeration must raise,
        # not silently shrink the corpus.
        def handler(request: httpx.Request) -> httpx.Response:
            page = CALENDAR_PAGE if "fomccalendars" in str(request.url) else HISTORICAL_PAGE
            return httpx.Response(200, text=page)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(RuntimeError, match="coverage gap"):
            fed.enumerate_statements(client=client, historical_years=(2008,))

    def test_all_url_eras_collected(self):
        refs = {ref.doc_id: ref for ref in self.enumerate()}
        assert set(refs) == {"fed_20250618a", "fed_20080122b", "fed_20000516", "fed_20050202"}

    def test_traps_rejected(self):
        urls = "".join(ref.url for ref in self.enumerate())
        assert "20250822a" not in urls  # notation-vote 'Statement on Longer-Run Goals'
        assert "a1" not in urls  # implementation notes (htm and pdf)
        assert "fomcminutes" not in urls  # minutes link with text 'HTML'
        assert "LongerRunGoals" not in urls  # historical PDF trap

    def test_dates_and_suffixes(self):
        refs = {ref.doc_id: ref for ref in self.enumerate()}
        assert refs["fed_20080122b"].announcement_date == date(2008, 1, 22)
        assert refs["fed_20000516"].announcement_date == date(2000, 5, 16)
        assert refs["fed_20250618a"].url.endswith("monetary20250618a.htm")


class TestExtractModern:
    def test_body_only_with_junk_stripped(self):
        text = fed.extract_text(MODERN_PAGE)
        assert text.startswith("Published: June 18, 2025")  # release date grounded
        assert "economic activity has continued to expand" in text
        assert "4-1/4 to 4 1/2 percent" in text  # NBSP normalized
        assert "media inquiries" not in text  # Cloudflare email paragraph
        assert "Implementation Note" not in text  # link-only paragraph
        assert "Federal Reserve issues FOMC statement" not in text  # heading div


class TestExtractBoarddocs:
    def test_slices_between_marker_comments(self):
        text = fed.extract_text(BOARDDOCS_PAGE)
        assert text.startswith("Release Date: November 6, 2002")  # date grounded
        assert "lower its target for the federal funds rate by 50 basis points" in text
        assert "reduction in the discount rate" in text
        assert "Last update" not in text
        assert "2002 Monetary policy" not in text

    def test_unclosed_p_tags_do_not_duplicate_text(self):
        # Legacy pages leave <p> unclosed; html.parser nests them, and naive
        # per-paragraph get_text() re-emits every following paragraph.
        text = fed.extract_text(BOARDDOCS_PAGE)
        assert text.count("reduction in the discount rate") == 1
        assert text.count("50 basis point") == 2  # once per genuine paragraph

    def test_fallback_without_marker_comments(self):
        page = BOARDDOCS_PAGE.replace("DO NOT REMOVE:  Wireless Generation", "gone")
        text = fed.extract_text(page)
        assert "federal funds rate by 50 basis points" in text
        assert "Last update" not in text

    def test_unrecognized_layout_raises(self):
        with pytest.raises(ValueError, match="unrecognized"):
            fed.extract_text("<html><body><p>nothing here</p></body></html>")


@pytest.mark.skipif(
    os.getenv("RATEGAUGE_LIVE", "").lower() not in {"1", "true", "yes"},
    reason="live federalreserve.gov test; set RATEGAUGE_LIVE=1 to run",
)
class TestLive:
    def test_enumeration_and_extraction_match_known_history(self):
        refs = fed.enumerate_statements()
        # ~224 statements 2000..2026 YTD (HF-dataset cross-check).
        assert 210 <= len(refs) <= 240
        by_id = {ref.doc_id: ref for ref in refs}
        assert "fed_20080122b" in by_id  # emergency cut, 'b' suffix
        assert "fed_20200315a" in by_id  # 2020 emergency cut (index links the 'a' variant)
        assert "fed_20260617a" in by_id  # latest as of 2026-07

        from rategauge.http import default_client

        with default_client(browser_headers=True) as client:
            modern = client.get(by_id["fed_20250618a"].url)
            modern.raise_for_status()
            legacy = client.get(by_id["fed_20021106"].url)
            legacy.raise_for_status()
        assert "economic activity" in fed.extract_text(modern.text)
        assert "50 basis points" in fed.extract_text(legacy.text)
