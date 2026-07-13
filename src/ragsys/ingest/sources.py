"""Fetch SEBI / RBI circulars.

Two modes:
  * `download_urls([...])`  -> download an explicit list of PDF URLs you supply.
  * `sample_corpus()`       -> a tiny built-in text corpus of paraphrased,
                               public-domain-style circular content so the whole
                               pipeline (ingest -> retrieve -> eval) runs end to
                               end with ZERO network access, keys, or scraping.

We deliberately do NOT scrape sebi.gov.in / rbi.org.in automatically: their
terms and rate limits vary. Point `download_urls` at PDFs you are permitted to
fetch, or drop PDFs into data/pdfs/ and run the ingest CLI on the folder.
"""
from __future__ import annotations

from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _get(url: str) -> bytes:
    headers = {"User-Agent": "ragsys-ingest/0.1 (+research)"}
    with httpx.Client(timeout=30, follow_redirects=True, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content


def download_urls(urls: list[str], dest: str | Path = "data/pdfs") -> list[Path]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for url in urls:
        name = url.rstrip("/").split("/")[-1] or "circular"
        if not name.lower().endswith(".pdf"):
            name += ".pdf"
        out = dest / name
        try:
            out.write_bytes(_get(url))
            saved.append(out)
            print(f"[sources] saved {url} -> {out}")
        except Exception as exc:  # keep going on individual failures
            print(f"[sources] FAILED {url}: {exc}")
    return saved


# ---------------------------------------------------------------------------
# Offline sample corpus. Paraphrased summaries of well-known circular topics so
# the demo is self-contained. Replace with real PDFs for production use.
# ---------------------------------------------------------------------------
SAMPLE_DOCS: list[dict] = [
    {
        "filename": "sebi_ria_2013.txt",
        "source": "SEBI",
        "circular_no": "SEBI/HO/IMD/DF1/CIR/2013/07",
        "title": "SEBI (Investment Advisers) Regulations",
        "text": (
            "1. These regulations govern the registration and conduct of Investment "
            "Advisers. No person shall act as an investment adviser without obtaining "
            "a certificate of registration from the Board, unless specifically exempt.\n\n"
            "2. Eligibility. An individual investment adviser must hold a professional "
            "qualification or post-graduate degree in finance, and a certification from "
            "NISM. A minimum net worth of Rupees five lakh is required for individuals "
            "and Rupees fifty lakh for non-individuals.\n\n"
            "3. Segregation. An investment adviser shall keep advisory activities "
            "segregated from distribution activities. The same person cannot provide "
            "advice and earn commission on the same product to the same client.\n\n"
            "4. Fees. An investment adviser may charge fees either as a percentage of "
            "assets under advice, capped at 2.5 percent of AUA per annum, or a fixed "
            "fee not exceeding Rupees one lakh twenty five thousand per annum per client."
        ),
    },
    {
        "filename": "sebi_mf_riskometer_2020.txt",
        "source": "SEBI",
        "circular_no": "SEBI/HO/IMD/DF3/CIR/2020/197",
        "title": "Product Labeling in Mutual Funds - Riskometer",
        "text": (
            "1. Every mutual fund scheme shall disclose a Riskometer with six levels of "
            "risk: Low, Low to Moderate, Moderate, Moderately High, High and Very High.\n\n"
            "2. The risk level of a scheme shall be evaluated on a monthly basis based on "
            "the portfolio as on the last day of the month, and disclosed within ten days "
            "from the close of each month on the AMC website and on AMFI.\n\n"
            "3. Any change in Riskometer shall be communicated to unit holders by way of a "
            "notice cum addendum and by email or SMS. The number of times the Riskometer "
            "changed during the year shall be disclosed in the annual report."
        ),
    },
    {
        "filename": "rbi_kyc_master_2016.txt",
        "source": "RBI",
        "circular_no": "RBI/2015-16/42",
        "title": "Master Direction - Know Your Customer (KYC) Direction",
        "text": (
            "1. Regulated entities shall frame their KYC policy incorporating Customer "
            "Acceptance Policy, Risk Management, Customer Identification Procedures and "
            "Monitoring of Transactions.\n\n"
            "2. Customer Due Diligence. At the time of commencement of an account-based "
            "relationship, regulated entities shall obtain a certified copy of an "
            "officially valid document containing details of identity and address, and a "
            "recent photograph.\n\n"
            "3. Periodic updation shall be carried out at least once in every two years "
            "for high risk customers, once in every eight years for medium risk and once "
            "in every ten years for low risk customers.\n\n"
            "4. Regulated entities shall apply enhanced due diligence measures for "
            "customers classified as high risk, including politically exposed persons."
        ),
    },
    {
        "filename": "rbi_digital_lending_2022.txt",
        "source": "RBI",
        "circular_no": "RBI/2022-23/111",
        "title": "Guidelines on Digital Lending",
        "text": (
            "1. All loan disbursals and repayments shall be executed only between the "
            "bank accounts of the borrower and the regulated entity, without any pass "
            "through or pool account of the lending service provider.\n\n"
            "2. Any fees payable to lending service providers shall be paid directly by "
            "the regulated entity and not by the borrower.\n\n"
            "3. A standardised Key Fact Statement must be provided to the borrower before "
            "execution of the contract, disclosing the all-inclusive Annual Percentage "
            "Rate and the recovery mechanism.\n\n"
            "4. Borrowers shall be given a cooling-off period during which they can exit "
            "the digital loan by paying the principal and proportionate APR without any "
            "penalty."
        ),
    },
    {
        "filename": "sebi_lodr_related_party_2021.txt",
        "source": "SEBI",
        "circular_no": "SEBI/HO/CFD/CMD1/CIR/2021/662",
        "title": "LODR - Related Party Transactions",
        "text": (
            "1. A related party transaction shall require prior approval of the audit "
            "committee. Only independent directors on the audit committee shall approve "
            "such transactions.\n\n"
            "2. With effect from 1 April 2022, a transaction with a related party shall be "
            "considered material if the value exceeds Rupees one thousand crore or ten "
            "percent of the annual consolidated turnover, whichever is lower.\n\n"
            "3. Material related party transactions shall require prior approval of the "
            "shareholders through a resolution, and no related party shall vote to approve "
            "such resolution whether the entity is a party to the transaction or not."
        ),
    },
]


def sample_corpus(dest: str | Path = "data/pdfs") -> list[Path]:
    """Write the offline sample corpus to disk and return the file paths."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for d in SAMPLE_DOCS:
        p = dest / d["filename"]
        header = f"{d['title']}\n{d['circular_no']}\n\n"
        p.write_text(header + d["text"], encoding="utf-8")
        paths.append(p)
    print(f"[sources] wrote {len(paths)} sample documents to {dest}")
    return paths
