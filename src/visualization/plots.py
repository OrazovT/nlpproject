from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "results" / "combined.csv"
PLOTS = PROJECT_ROOT / "results" / "plots"

def load_scores(path:Path) -> pd.DataFrame:
    return pd.read_csv(path)

def plot_tfidf_histogram(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df["tfidf_score"], bins=20)
    plt.xlabel("TF-IDF similarity score")
    plt.ylabel("Number of papers")
    plt.title("Distribution of TF-IDF")
    plt.tight_layout()

    output_path = PLOTS / "tfidf_histogram.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"saved {output_path}")

def plot_embedding_histogram(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df["embedding_score"], bins=20)
    plt.xlabel("Embedding similarity score")
    plt.ylabel("Number of papers")
    plt.title("Distribution of Embedding")
    plt.tight_layout()

    output_path = PLOTS / "embedding_histogram.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"saved {output_path}")

def plot_score_comparison(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(df["tfidf_score"], df["embedding_score"])
    plt.xlabel("TF-IDF similarity score")
    plt.ylabel("Embedding similarity score")
    plt.title("TF-IDF vs Embedding scores")
    plt.tight_layout()

    output_path = PLOTS / "score_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"saved {output_path}")

if __name__ == "__main__":
    PLOTS.mkdir(parents=True, exist_ok=True)
    scores = load_scores(INPUT_PATH)

plot_tfidf_histogram(scores)
plot_embedding_histogram(scores)
plot_score_comparison(scores)
print("All plots created")