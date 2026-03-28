"""
STAGE 2: Sentiment Analysis
Runs sentiment scoring on Reddit text + optionally travel blog text.
Tags each record by inferred country/region for cross-cultural comparison.

Install: pip install pandas transformers torch scikit-learn matplotlib seaborn
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

# ── Config ────────────────────────────────────────────────────────────────────

# Using a pretrained model from HuggingFace — no training needed.
# cardiffnlp/twitter-roberta-base-sentiment-latest is good for social text.
# Alternative: "distilbert-base-uncased-finetuned-sst-2-english" (simpler, faster)
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Country/region keyword mapping for tagging text to a region
# Applied to post title + text + subreddit
COUNTRY_KEYWORDS = {
    "japan":     ["japan", "japanese", "tokyo", "osaka", "kyoto", "nippon",
                  "manhole card", "pokemon lid", "pokefuta", "マンホール"],
    "singapore": ["singapore", "sg", "singaporean"],
    "uk":        ["london", "uk", "united kingdom", "britain", "england"],
    "usa":       ["new york", "nyc", "usa", "america", "american", "us "],
    "germany":   ["germany", "german", "berlin", "kanaldeckel"],
    "france":    ["france", "paris", "french"],
    "india":     ["india", "indian", "mumbai", "delhi"],
    "generic":   ["manhole", "drain cover", "sewer cover"],  # fallback
}

# ── Country Tagger ────────────────────────────────────────────────────────────

def infer_country(row):
    """Guess which country a post is about from its text content."""
    text = f"{row.get('title','')} {row.get('text','')} {row.get('subreddit','')}".lower()

    for country, keywords in COUNTRY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return country

    return "unknown"


# ── Sentiment Scorer ──────────────────────────────────────────────────────────

def load_sentiment_model():
    print(f"Loading model: {SENTIMENT_MODEL}...")
    return pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        max_length=512,
        truncation=True,
    )


def score_texts(df, sentiment_pipe):
    """
    Run sentiment on the 'text' column.
    Returns scores mapped to: positive=1, neutral=0, negative=-1
    """
    label_map = {
        "positive": 1, "POSITIVE": 1, "LABEL_2": 1,
        "neutral":  0, "NEUTRAL":  0, "LABEL_1": 0,
        "negative": -1,"NEGATIVE":-1, "LABEL_0": -1,
    }

    texts = df["text"].fillna("").tolist()
    results = []

    # Batch in groups of 32 to avoid memory issues
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            preds = sentiment_pipe(batch)
            for pred in preds:
                results.append({
                    "label":       pred["label"],
                    "confidence":  pred["score"],
                    "sentiment":   label_map.get(pred["label"], 0),
                })
        except Exception as e:
            print(f"  ⚠ Batch {i//batch_size} failed: {e}")
            results.extend([{"label":"error","confidence":0,"sentiment":0}] * len(batch))

        if i % 320 == 0:
            print(f"  {i}/{len(texts)} processed...")

    return pd.DataFrame(results)


# ── Weighted Scoring ──────────────────────────────────────────────────────────
# Upvote score on Reddit is a proxy for community agreement.
# Weight sentiment by score so a +500 comment counts more than a +1 comment.

def compute_weighted_sentiment(df):
    """Compute score-weighted mean sentiment per country."""

    # Clip score to avoid outliers dominating (log scale)
    df["score_weight"] = (df["score"].clip(lower=1)).apply(lambda x: x**0.5)

    # Weighted average
    def wavg(group):
        return (group["sentiment"] * group["score_weight"]).sum() / group["score_weight"].sum()

    by_country = df.groupby("country").apply(wavg).rename("weighted_sentiment")
    raw_counts = df.groupby("country").agg(
        n_posts=("sentiment", "count"),
        mean_sentiment=("sentiment", "mean"),
        pct_positive=("sentiment", lambda x: (x == 1).mean() * 100),
        pct_negative=("sentiment", lambda x: (x == -1).mean() * 100),
    )

    return pd.concat([by_country, raw_counts], axis=1).sort_values(
        "weighted_sentiment", ascending=False
    )


# ── Keyword Analysis ──────────────────────────────────────────────────────────
# What WORDS do people use when talking about manholes in each country?
# This is your qualitative design vocabulary.

DESIGN_VOCAB = {
    "aesthetic":  ["beautiful", "art", "gorgeous", "stunning", "lovely", "pretty",
                   "artistic", "elegant", "intricate", "detailed"],
    "functional": ["safe", "functional", "practical", "sturdy", "durable",
                   "slippery", "dangerous", "hazard", "trip", "broken"],
    "cultural":   ["culture", "tradition", "local", "pride", "identity",
                   "community", "landmark", "heritage", "unique", "symbol"],
    "negative":   ["ugly", "boring", "dull", "plain", "generic", "eyesore",
                   "dirty", "rusted", "neglected", "terrible"],
    "engagement": ["collect", "photograph", "hunt", "find", "discover", "tourist",
                   "attraction", "visit", "seek", "card"],
}

def keyword_frequency_by_country(df):
    """Count design vocabulary hits per country — shows what people notice."""
    records = []
    for country, group in df.groupby("country"):
        text_blob = " ".join(group["text"].fillna("").str.lower())
        row = {"country": country}
        for category, words in DESIGN_VOCAB.items():
            row[category] = sum(text_blob.count(w) for w in words)
        records.append(row)
    return pd.DataFrame(records).set_index("country")


# ── Visualizations ────────────────────────────────────────────────────────────

def plot_sentiment_by_country(summary_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in summary_df["weighted_sentiment"]]
    summary_df["weighted_sentiment"].plot(
        kind="barh", ax=ax, color=colors, edgecolor="black", linewidth=0.5
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Weighted Sentiment Score by Country\n(Reddit posts + comments)", fontsize=13)
    ax.set_xlabel("Weighted Sentiment Score (−1 to +1)")
    plt.tight_layout()
    plt.savefig("output/sentiment_by_country.png", dpi=150)
    plt.show()


def plot_keyword_heatmap(kw_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    # Normalize per row so countries with more text don't dominate
    kw_norm = kw_df.div(kw_df.sum(axis=1), axis=0)
    sns.heatmap(kw_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=ax)
    ax.set_title("Design Vocabulary Distribution by Country\n(normalized)", fontsize=13)
    plt.tight_layout()
    plt.savefig("output/keyword_heatmap.png", dpi=150)
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    import os
    os.makedirs("output", exist_ok=True)

    # Load scraped data
    df = pd.read_csv("/data/text_raw.csv")
    print(f"Loaded {len(df)} text records")

    # Tag countries
    df["country"] = df.apply(infer_country, axis=1)
    print(f"Country distribution:\n{df['country'].value_counts().to_string()}\n")

    # Drop unknowns (can't compare them)
    df = df[df["country"] != "unknown"]

    # Run sentiment
    sentiment_pipe = load_sentiment_model()
    print(f"\nScoring sentiment on {len(df)} records...")
    scores_df = score_texts(df, sentiment_pipe)
    df = pd.concat([df.reset_index(drop=True), scores_df], axis=1)

    # Save intermediate
    df.to_csv("/data/text_with_sentiment.csv", index=False)

    # Aggregate
    print("\n=== Sentiment Summary ===")
    summary = compute_weighted_sentiment(df)
    print(summary.to_string())
    summary.to_csv("output/sentiment_summary.csv")

    # Keyword analysis
    print("\n=== Design Vocabulary ===")
    kw_df = keyword_frequency_by_country(df)
    print(kw_df.to_string())
    kw_df.to_csv("output/keyword_frequency.csv")

    # Plots
    plot_sentiment_by_country(summary)
    plot_keyword_heatmap(kw_df)

    print("\n✓ Sentiment analysis complete")


if __name__ == "__main__":
    run()
