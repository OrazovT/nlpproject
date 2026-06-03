from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TFIDF_PATH = PROJECT_ROOT / "results" / "tfidf_scores.csv"
EMBEDDING_PATH = PROJECT_ROOT / "results" / "embedding_scores.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "combined.csv"

def combine_scores(tfidf_df: pd.DataFrame, embedding_df: pd.DataFrame) -> pd.DataFrame:
    embedding_scores = embedding_df[["title", "year", "url", "embedding_score"]]
    combined_df = tfidf_df.merge(
        embedding_scores,
        on = ["title", "year", "url"],
        how="inner"
    )
    return combined_df

def print_summary(combined_df: pd.DataFrame) -> None:
    
    print("number of papers:", len(combined_df))

    print("\nTF-IDF:")
    print(combined_df["tfidf_score"].describe())
    print("\nembedding:")
    print(combined_df["embedding_score"].describe())

    correlation = combined_df["tfidf_score"].corr(combined_df["embedding_score"])

    print("\ncorrelation:")
    print(correlation)

    print("\nlowest embedding papers:")
    print(
        combined_df
        .sort_values(by="embedding_score")
        [["title", "year", "tfidf_score", "embedding_score"]]
        .head(10)
    )

if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tfidf = pd.read_csv(TFIDF_PATH)
    embeddings = pd.read_csv(EMBEDDING_PATH)
    combined = combine_scores(tfidf, embeddings)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"saved to {OUTPUT_PATH}")
    print_summary(combined)