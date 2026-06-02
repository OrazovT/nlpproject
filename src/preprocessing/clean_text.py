from pathlib import Path
import pandas as pd
import re
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "jmlr_papers.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "papers_clean.csv"
def normalize_whitespace(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title"] = df["title"].apply(normalize_whitespace)
    df["abstract"] = df["abstract"].apply(normalize_whitespace)

    if "url" in df.columns:
        df["url"] =df["url"].apply(normalize_whitespace)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df[df["title"] != ""]
    df = df[df["abstract"] != ""]
    df = df.drop_duplicates(subset=["title", "year"])
    df = df.sort_values(by="year", ascending=False)
    df["abstract_length"] = df["abstract"].str.len()
    return df
if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok="True")
    raw_df = pd.read_csv(INPUT_PATH)
    clean_df = clean_dataframe(raw_df)
    clean_df.to_csv(OUTPUT_PATH, index=False)
    print(f"loaded {len(raw_df)}")
    print(f"saved {len(clean_df)} to {OUTPUT_PATH}")
    print(clean_df[["title", "year", "abstract_length"]].head())