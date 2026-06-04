from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "results" / "combined.csv"
LOW_EMBEDDING_PATH = PROJECT_ROOT / "results" / "low_embedding.csv"
LOW_TFIDF_PATH = PROJECT_ROOT / "results" / "low_tfidf.csv"
DISAGREEMENT_PATH = PROJECT_ROOT / "results" / "method_disagreement.csv"

def load_scores(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def get_low_embedding_papers(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        df.sort_values(by="embedding_score")
        [["title", "year", "tfidf_score", "embedding_score", "url", "abstract"]]
        .head(n)    
    )

def get_low_tfidf_papers(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        df.sort_values(by="tfidf_score")
        [["title", "year", "tfidf_score", "embedding_score", "url", "abstract"]]
        .head(n)  
    )

def get_method_disagreements(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    result = df.copy()
    result["embedding_rank"] = result["embedding_score"].rank(ascending=True)
    result["tfidf_rank"] = result["tfidf_score"].rank(ascending=True)
    result["rank_diff"] = (
        result["tfidf_rank"] - result["embedding_rank"]
        ).abs()
    return(
        result.sort_values(by="rank_diff", ascending=True)
        [["title", "year", "tfidf_score", "embedding_score", "rank_diff", "url", "abstract"]]
    ).head(n)

if __name__ == "__main__":
    scores = load_scores(INPUT_PATH)
    low_embedding = get_low_embedding_papers(scores)
    low_tfidf = get_low_tfidf_papers(scores)
    disagreements = get_method_disagreements(scores)

    low_embedding.to_csv(LOW_EMBEDDING_PATH, index=False)
    low_tfidf.to_csv(LOW_TFIDF_PATH, index=False)
    disagreements.to_csv(DISAGREEMENT_PATH, index=False)

    print(f"saved low embedding papers to {LOW_EMBEDDING_PATH}")
    print(f"saved low TF-IDF papers to {LOW_TFIDF_PATH}")
    print(f"saved disagreements to {DISAGREEMENT_PATH}")
    print("\nLowest embedding alignment papers:")
    print(low_embedding[["title", "year", "url", "tfidf_score", "embedding_score"]].head())