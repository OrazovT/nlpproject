from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AIMS_SCOPE_PATH = PROJECT_ROOT / "data" / "raw" / "aims_scope.txt"
PAPERS_PATH = PROJECT_ROOT / "data" / "processed" / "papers_clean.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "tfidf_scores.csv"

def load_aims_scope(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def compute_tfidf_alignment(aims_scope: str, papers_df: pd.DataFrame) -> pd.DataFrame:
    texts = [aims_scope] + papers_df["abstract"].tolist()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    aims_vector = tfidf_matrix[0]
    abstracts_vectors = tfidf_matrix[1:]
    scores = cosine_similarity(abstracts_vectors, aims_vector).flatten()
    result_df = papers_df.copy()
    result_df["tfidf_score"] = scores
    return result_df
if __name__ =="__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    aims_scope_text = load_aims_scope(AIMS_SCOPE_PATH)
    papers = pd.read_csv(PAPERS_PATH)
    result = compute_tfidf_alignment(aims_scope_text, papers)
    result.to_csv(OUTPUT_PATH, index = False)
    print(f"saved to {OUTPUT_PATH}")
    print(result[["title", "year", "tfidf_score"]].head())
    print(result["tfidf_score"].describe())