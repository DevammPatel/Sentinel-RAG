import json

from ragsys.ingest.loader import _clean, _extract_meta, html_to_text, load_any
from ragsys.ingest.pipeline import load_path

RBI_PAGE = """
<html><body><form id="aspnet">
<table class="tablebg"><tr><td>Home</td></tr></table>
<table class="tablebg"><tr><td>
  <p>Reserve Bank of India (Commercial Banks - Credit Cards) Directions, 2025</p>
  <p>RBI/DOR/2025-26/155<br>DOR.AUT.REC.No.74/24-01-041/2025-26</p>
  <p>November 28, 2025</p>
  <p>19. Any request for closure of a credit card shall be honoured within seven working
     days by the credit card-issuer.</p>
  <p>20. If a credit card has not been used for more than one year, closure shall begin.</p>
</td></tr></table>
<script>var tracking = "should not appear";</script>
</form></body></html>
"""


def test_html_body_inside_a_form_is_kept():
    # ASP.NET pages wrap everything in <form>; dropping forms would empty the page.
    text = html_to_text(RBI_PAGE)
    assert "seven working days" in text
    assert "should not appear" not in text


def test_html_block_elements_become_separate_lines():
    lines = [ln.strip() for ln in _clean(html_to_text(RBI_PAGE)).splitlines() if ln.strip()]
    assert any(ln.startswith("19. Any request") for ln in lines)
    assert any(ln.startswith("20. If a credit card") for ln in lines)


def test_rbi_reference_and_date():
    meta = _extract_meta("Directions, 2025\n\nRBI/DOR/2025-26/155\nDOR.AUT\n\nNovember 28, 2025\n")
    assert meta["circular_no"] == "RBI/DOR/2025-26/155"
    assert meta["issued_on_raw"] == "November 28, 2025"


def test_rbi_reference_with_mixed_case_department():
    assert _extract_meta("RBI/DoS/2026-27/412\nJuly 31, 2026")["circular_no"] == "RBI/DoS/2026-27/412"


def test_sebi_2026_reference_format():
    meta = _extract_meta("CIRCULAR\n\nHO/17/11/22(1)2026-DDHS-POD1/ I/23092/2026 October 07, 2026\n\nTo,")
    assert meta["circular_no"] == "HO/17/11/22(1)2026-DDHS-POD1/I/23092/2026"
    assert meta["issued_on_raw"] == "October 07, 2026"


def test_sebi_legacy_reference_format():
    meta = _extract_meta("CIRCULAR\nSEBI/HO/MIRSD/MIRSD-PoD-1/P/CIR/2023/73 May 19, 2023")
    assert meta["circular_no"] == "SEBI/HO/MIRSD/MIRSD-PoD-1/P/CIR/2023/73"


def test_date_does_not_swallow_digits_from_the_reference_number():
    # "…/2025-26\n\nNovember 28, 2025" used to be read as "26 November 28, 20".
    assert _extract_meta("RBI/DOR/2025-26\n\nNovember 28, 2025")["issued_on_raw"] == "November 28, 2025"


def test_numeric_date_format():
    assert _extract_meta("HO/(449)2026-ITD-5_DIV1/I/19448/2026 24.08.2026")["issued_on_raw"] == "24.08.2026"


def test_reference_cited_deep_in_the_body_is_ignored():
    body = "Some circular without a number at the top.\n" + "filler text. " * 80
    body += "\nSEBI vide circular SEBI/HO/MRD/CIR/P/2024/00181 specified otherwise."
    assert "circular_no" not in _extract_meta(body)


def test_clean_drops_page_furniture_and_hindi_lines():
    raw = "Page 1 of 8\nCIRCULAR\nसभी डिपॉजिटरी\nAll Depositories\n(438 kb)\nPrevious Versions\nreg-\nulation"
    cleaned = _clean(raw)
    assert "Page 1 of 8" not in cleaned
    assert "डिपॉजिटरी" not in cleaned
    assert "438 kb" not in cleaned and "Previous Versions" not in cleaned
    assert "All Depositories" in cleaned
    assert "regulation" in cleaned  # hyphenated line break rejoined


def test_manifest_supplies_title_url_and_stable_id(tmp_path):
    (tmp_path / "rbi-1.html").write_text(RBI_PAGE, encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "rbi-1.html": {"id": "rbi-1", "regulator": "RBI", "title": "Card Directions",
                       "url": "https://www.rbi.org.in/x"}
    }), encoding="utf-8")
    (doc,) = load_path(tmp_path)  # manifest.json itself must not be ingested
    assert (doc.doc_id, doc.title, doc.source) == ("rbi-1", "Card Directions", "RBI")
    assert doc.url == "https://www.rbi.org.in/x"
    assert doc.circular_no == "RBI/DOR/2025-26/155"


def test_text_file_without_manifest_guesses_metadata(tmp_path):
    f = tmp_path / "sebi_note.txt"
    f.write_text("Securities and Exchange Board of India\nSome guidance text.", encoding="utf-8")
    doc = load_any(f)
    assert doc.source == "SEBI"
    assert doc.doc_id == "sebi-note"
    assert doc.title == "Securities and Exchange Board of India"


def test_pdf_lines_wrapped_mid_sentence_are_rejoined():
    from ragsys.ingest.loader import unwrap_lines

    raw = (
        "To,\n"
        "All Stock Exchanges\n"
        "All Depositories\n"
        "\n"
        "1. With an objective to ease and expedite the process of accreditation, the SEBI\n"
        "(Alternative Investment Funds) Regulations, 2012 have been amended and notified on\n"
        "October 15, 2026.\n"
        "2. The modalities for implementing these provisions are specified below in detail\n"
        "3.1. A Manager may determine and record an investor's accredited status.\n"
    )
    lines = unwrap_lines(raw).split("\n")
    assert lines[:3] == ["To,", "All Stock Exchanges", "All Depositories"]  # short lines kept
    assert lines[4].startswith("1. With an objective") and lines[4].endswith("October 15, 2026.")
    # A long line is not glued to the next one when that line opens a new clause.
    assert lines[5].startswith("2. The modalities") and lines[5].endswith("in detail")
    assert lines[6].startswith("3.1. A Manager")


def test_unwrap_adapts_to_narrow_two_column_layouts():
    from ragsys.ingest.loader import unwrap_lines

    raw = ("1. With the rapid pace of technological\n"
           "developments in securities market, the\n"
           "frequency and sophistication of cyber\n"
           "incidents are also on the rise.\n")
    assert unwrap_lines(raw).strip() == (
        "1. With the rapid pace of technological developments in securities market, the "
        "frequency and sophistication of cyber incidents are also on the rise."
    )
