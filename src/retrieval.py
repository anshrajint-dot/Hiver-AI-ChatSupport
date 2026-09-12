from pathlib import Path
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent.parent

THREADS_FILE = BASE_DIR / "data" / "processed" / "applesupport_threads.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "resolution_index.csv"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_index():

    print("=" * 60)
    print("BUILDING HISTORICAL RESOLUTION INDEX")
    print("=" * 60)

    df = pd.read_csv(THREADS_FILE)

    print(f"Historical conversations: {len(df)}")

    df = df.dropna(subset=["customer_text", "support_reply"])

    df["customer_text"] = df["customer_text"].astype(str)
    df["support_reply"] = df["support_reply"].astype(str)

    # Remove empty replies
    df = df[
        (df["customer_text"].str.strip() != "") &
        (df["support_reply"].str.strip() != "")
    ]

    # Clean customer messages for retrieval
    df["customer_clean"] = df["customer_text"].apply(clean_text)

    # Remove duplicate customer/reply pairs
    df = df.drop_duplicates(
        subset=["customer_clean", "support_reply"]
    )

    # Keep useful columns
    index_df = df[
        [
            "customer_tweet_id",
            "customer_text",
            "support_tweet_id",
            "support_reply",
            "customer_time",
            "support_time",
            "customer_clean",
        ]
    ].copy()

    index_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print("RESOLUTION INDEX CREATED")
    print("=" * 60)

    print(f"Resolution examples: {len(index_df)}")
    print(f"Saved to: {OUTPUT_FILE}")


class HistoricalRetriever:

    def __init__(self, index_file=OUTPUT_FILE):

        self.df = pd.read_csv(index_file)

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True
        )

        self.matrix = self.vectorizer.fit_transform(
            self.df["customer_clean"].fillna("")
        )

    def search(self, query, top_k=5):

        query_clean = clean_text(query)

        query_vector = self.vectorizer.transform(
            [query_clean]
        )

        scores = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        top_indices = scores.argsort()[-top_k:][::-1]

        results = []

        for idx in top_indices:

            results.append({
                "customer_text": self.df.iloc[idx]["customer_text"],
                "support_reply": self.df.iloc[idx]["support_reply"],
                "similarity": round(float(scores[idx]), 4)
            })

        return results


if __name__ == "__main__":
    build_index()

    print()
    print("Testing retrieval...")

    retriever = HistoricalRetriever()

    test_query = (
        "My iPhone battery drains very quickly "
        "after the latest iOS update"
    )

    results = retriever.search(
        test_query,
        top_k=5
    )

    print()
    print("QUERY:")
    print(test_query)

    print()
    print("TOP HISTORICAL RESOLUTIONS:")
    print("-" * 60)

    for i, result in enumerate(results, 1):

        print(f"\n{i}. Similarity: {result['similarity']}")

        print("Customer:")
        print(result["customer_text"])

        print("AppleSupport:")
        print(result["support_reply"])