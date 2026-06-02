from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT =  Path(__file__).resolve().parents[2]
AIMS_SCOPE_PATH = PROJECT_ROOT / "data" / "raw" / "aims_scope.txt"
PAPERS_PATH = PROJECT_ROOT / "data" / "processed" / "papers_clean.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "embedding_scores.csv"
MODEL_NAME = "all-MiniLM-L6-v2"

def load_aims_scope(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def compute_embedding_allignment(aims_scope:str, papers_df: pd.DataFrame) -> pd.DataFrame:
    model = SentenceTransformer(MODEL_NAME)
    texts =[aims_scope] + papers_df["abstract"].tolist()
    embeddings = model.encode(
        texts,
        convert_to_numpy = True,
        show_progress_bar=True
    )
    
    aims_embedding = embeddings[0].reshape(1, -1)
    abstract_embeddings = embeddings[1:]
    scores = cosine_similarity(abstract_embeddings, aims_embedding).flatten()
    result_df = papers_df.copy()
    result_df["embedding_score"] = scores
    return result_df

if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    aims_scope_texts = load_aims_scope(AIMS_SCOPE_PATH)
    papers = pd.read_csv(PAPERS_PATH)
    result = compute_embedding_allignment(aims_scope_texts, papers)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"saved to {OUTPUT_PATH}")
    print(result[["title", "year", "embedding_score"]].head())
    print(result["embedding_score"].describe())