from pathlib import Path
import pandas as pd
import requests
import time
import re 
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "data" / "raw" / "jmlr_papers.csv"
BASE_URL = "https://www.jmlr.org"
VOLUME_URLS = [
    "https://www.jmlr.org/papers/v25/",
    "https://www.jmlr.org/papers/v24/",
    "https://www.jmlr.org/papers/v23/"
]
SLEEP_SEC = 1
MAX_PAPERS = 80 

def get_html(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text

def extract_abs_links(volume_url: str) -> list[str]:
    html = get_html(volume_url)
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a_tag in soup.find_all("a"):
        href = a_tag.get("href", "")
        text = a_tag.get_text(strip=True).lower()

        if text == "abs" and href:
            full_url = requests.compat.urljoin(volume_url, href)
            links.append(full_url)
    return links

def extract_paper_data(abs_url: str) -> dict:
    html = get_html(abs_url)
    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text("\n", strip=True)
    
    title = ""
    meta_title = soup.find("meta", attrs={"name": "citation_title"})
    if meta_title and meta_title.get("content"):
        title = meta_title["content"].strip()
    if not title:
        page_title = soup.find("title")
        if page_title:
            title = page_title.get_text(" ", strip=True)
    if not title:
        heading = soup.find(["h1", "h2", "h3"])
        if heading: title = heading.get_text(" ", strip=True)
    year_match = re.search(r"\b(20\d{2})\b", page_text)
    year = int(year_match.group(1)) if year_match else None

    abstract = ""
    abstract_heading = soup.find(
        lambda tag: tag.name in ["h2", "h3", "b", "strong"]
        and "abstract" in tag.get_text(strip=True).lower()
    )
    if abstract_heading:
        parts = []
        for sibling in abstract_heading.find_next_siblings():
            text = sibling.get_text(" ", strip=True)
            if not text:
                continue
            if sibling.name in ["h2", "h3"] and "abstract" not in text.lower():
                break
            parts.append(text)
        abstract = " ".join(parts)
    return {
        "title": title,
        "abstract": abstract,
        "year": year,
        "url": abs_url,
    }

def collect_papers() -> pd.DataFrame:
    rows = []
    for volume_url in VOLUME_URLS:
        print(f"volume: {volume_url}")
        abs_links = extract_abs_links(volume_url)
        print(f"found {len(abs_links)} abstracts")

        for abs_url in abs_links:
            if len(rows) >= MAX_PAPERS:
                break
            print(f"reading: {abs_url}")
            paper_data = extract_paper_data(abs_url)
            if paper_data["abstract"]:
                rows.append(paper_data)
            time.sleep(SLEEP_SEC)
        if len(rows) >= MAX_PAPERS:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        print("nothing collected")
        return df
    df = df.drop_duplicates(subset=["title", "url"])
    df = df.sort_values(by="year", ascending=False)
    return df

if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    papers_df = collect_papers()
    papers_df.to_csv(OUT_PATH, index=False)
    print(f"saved {len(papers_df)} to {OUT_PATH}")