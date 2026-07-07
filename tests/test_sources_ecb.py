"""Unit tests for ECB enumeration and extraction (network-free).

Fixtures replicate verbatim DOM quirks observed live on 2026-07-07: nested
'Related' accordions reusing the dt/dd pattern for accounts and PDFs,
language-selector duplicate hrefs, headerless fragments, era-specific body
layouts (1999 pages split text across section/orderedlist blocks).
"""

import os
from datetime import date

import httpx
import pytest

from rategauge.sources import ecb

FRAGMENT = """
<dt isoDate="2026-06-11"><div class="date">11 June 2026</div></dt>
<dd><div class="title"><a href="/press/pr/date/2026/html/ecb.mp260611~4d41bd5e83.en.html"  >Monetary policy decisions</a></div>
<div class="ecb-langSelector"><span class="offeredLanguage"><a  class='arrow' lang="en" href="/press/pr/date/2026/html/ecb.mp260611~4d41bd5e83.en.html"><span class="ecb-full">English</span></a></span>
<div class="moreLanguages"><div class="ecb-langPopup"><div class="otherlang"><a lang="bg" href="/press/pr/date/2026/html/ecb.mp260611~4d41bd5e83.bg.html">Other language</a></div></div></div></div>
<div class='accordion'><div class="header"><div class="title">Related</div></div><div class="content-box"><div class="definition-list "><dl>
<dt isoDate="2026-04-16"><div class="date">16 April 2026</div></dt>
<dd><div class="title"><a href="/press/accounts/2026/html/ecb.mg260416~6a27b0c258.en.html"  >Meeting of 18-19 March 2026</a></div></dd>
<dt isoDate="2026-04-30"><div class="date">30 April 2026</div></dt>
<dd><div class="title"><a href="/press/press_conference/monetary-policy-statement/shared/pdf/ecb.ds260430~1c397fa90c.en.pdf?642b"  >Combined monetary policy decisions and statement</a></div></dd>
</dl></div></div></div>
</dd>
<dt isoDate="2005-08-04"><div class="date">4 August 2005</div></dt>
<dd><div class="title"><a href="/press/pr/date/2005/html/pr050804_2.en.html"  >Monetary policy decisions</a></div></dd>
"""

MODERN_PAGE = """<html><body><nav><div class="section">CHROME JUNK</div></nav><main >
<div class="title"><ul><li>PRESS RELEASE</li></ul><h1>Monetary policy decisions</h1></div>
<div class="section"><p class="ecb-publicationDate">11 June 2026 </p>
<p>The Governing Council decided to raise the three key ECB interest rates by 25 basis points.</p>
<h2>Key ECB interest rates</h2>
<p>The rate on the deposit facility will be increased to 2.25%, with effect from 17 June 2026.</p>
<p>***</p>
<p>The President of the ECB will comment at a press conference.</p></div>
<div class="related-topics"><h4>Related topics</h4><ul><li><a class="taxonomy-tag">Key ECB interest rates</a></li></ul></div>
<div class="address-box -top-arrow"><h2>European Central Bank</h2></div></main></body></html>"""

PAGE_1999 = """<html><body><main >
<div class="ecb-pressCategory">PRESS RELEASE</div>
<div class="title"><h1 class="ecb-pressContentTitle">Monetary policy decisions</h1></div>
<div class="ecb-pressContentPubDate">2 December 1999</div>
<div class="section"><p>At today's meeting the Governing Council of the ECB took the following monetary policy decisions:</p></div>
<div class="orderedlist"><ol><li>The interest rates on the main refinancing operations will remain unchanged at 3.0%, 4.0% and 2.0% respectively.</li>
<li>The reference value for M3 growth will remain 4½%.</li></ol></div>
<div class="section"><p>The President of the ECB will comment on the considerations underlying these decisions.</p></div>
<div class="address-box -top-arrow"></div></main></body></html>"""


class TestEnumerate:
    def enumerate(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "index_include.en.html" in str(request.url)
            return httpx.Response(200, text=FRAGMENT)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        return ecb.enumerate_decisions(client=client, first_year=2026, last_year=2026)

    def test_keeps_only_decision_releases(self):
        refs = self.enumerate()
        assert [ref.doc_id for ref in refs] == ["ecb_pr050804_2", "ecb_mp260611"]

    def test_rejects_accounts_pdfs_and_language_duplicates(self):
        urls = "".join(ref.url for ref in self.enumerate())
        assert "accounts" not in urls
        assert ".pdf" not in urls
        assert ".bg.html" not in urls

    def test_dates_come_from_isodate_attribute(self):
        refs = {ref.doc_id: ref for ref in self.enumerate()}
        assert refs["ecb_mp260611"].announcement_date == date(2026, 6, 11)
        assert refs["ecb_pr050804_2"].announcement_date == date(2005, 8, 4)

    def test_urls_are_absolute(self):
        for ref in self.enumerate():
            assert ref.url.startswith("https://www.ecb.europa.eu/press/pr/date/")


class TestDateFromHref:
    def test_1999_century_pivot(self):
        assert ecb._date_from_href("/press/pr/date/1999/html/pr991202.en.html") == date(
            1999, 12, 2
        )

    def test_modern(self):
        assert ecb._date_from_href(
            "/press/pr/date/2026/html/ecb.mp260611~4d41bd5e83.en.html"
        ) == date(2026, 6, 11)


class TestExtract:
    def test_modern_page_body_only(self):
        text = ecb.extract_text(MODERN_PAGE)
        assert "raise the three key ECB interest rates" in text
        assert "with effect from 17 June 2026" in text
        assert "Key ECB interest rates" in text  # section heading kept as context
        assert text.startswith("Published: 11 June 2026")  # announcement date grounded
        assert text.count("11 June 2026") == 1  # ...and not duplicated in the body
        assert "***" not in text
        assert "European Central Bank" not in text  # address box
        assert "Related topics" not in text
        assert "CHROME JUNK" not in text  # div.section outside <main>

    def test_1999_page_keeps_orderedlist_blocks(self):
        text = ecb.extract_text(PAGE_1999)
        assert "took the following monetary policy decisions" in text
        assert "remain unchanged at 3.0%, 4.0% and 2.0%" in text  # from div.orderedlist
        assert "The President of the ECB will comment" in text  # second div.section
        assert text.startswith("Published: 2 December 1999")  # sibling pub-date div
        assert "PRESS RELEASE" not in text

    def test_page_without_main_raises(self):
        with pytest.raises(ValueError, match="no <main>"):
            ecb.extract_text("<html><body><div class='section'>x</div></body></html>")


@pytest.mark.skipif(
    os.getenv("RATEGAUGE_LIVE", "").lower() not in {"1", "true", "yes"},
    reason="live ECB website test; set RATEGAUGE_LIVE=1 to run",
)
class TestLive:
    def test_enumeration_and_extraction_match_known_history(self):
        refs = ecb.enumerate_decisions()
        # Snapshot 2026-07: 326 decision releases 1999..2026 YTD.
        assert 320 <= len(refs) <= 345
        by_id = {ref.doc_id: ref for ref in refs}
        assert by_id["ecb_mp260611"].announcement_date == date(2026, 6, 11)
        assert "ecb_pr991202" in by_id
        assert "ecb_pr050804_2" in by_id

        from rategauge.http import default_client

        with default_client(browser_headers=True) as client:
            response = client.get(by_id["ecb_mp260611"].url)
            response.raise_for_status()
        text = ecb.extract_text(response.text)
        assert "raise the three key ECB interest rates by 25 basis points" in text
        assert "with effect from 17 June 2026" in text
