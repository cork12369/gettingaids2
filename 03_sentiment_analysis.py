"""
STAGE 2: Sentiment Analysis
Runs sentiment scoring on text data + generates comprehensive visualizations.
Tags each record by inferred country/region for cross-cultural comparison.

Install: pip install pandas transformers torch scikit-learn matplotlib seaborn
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

# Using a pretrained model from HuggingFace — no training needed.
# cardiffnlp/twitter-roberta-base-sentiment-latest is good for social text.
# Alternative: "distilbert-base-uncased-finetuned-sst-2-english" (simpler, faster)
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Support both Zeabur (/data) and local development (./data)
_data_root = Path("/data") if Path("/data").exists() else Path("./data")
OUTPUT_DIR = _data_root / "output"
TEXT_CSV = _data_root / "text_raw.csv"
SENTIMENT_CSV = _data_root / "text_with_sentiment.csv"

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
    "italy":     ["italy", "italian", "rome", "milan", "roma", "milano"],
    "spain":     ["spain", "spanish", "madrid", "barcelona"],
    "australia": ["australia", "australian", "sydney", "melbourne"],
    "canada":    ["canada", "canadian", "toronto", "vancouver"],
    "brazil":    ["brazil", "brazilian", "sao paulo", "rio"],
    "netherlands": ["netherlands", "dutch", "amsterdam", "holland"],
    "south_korea": ["korea", "korean", "seoul", "south korea"],
    "thailand":  ["thailand", "thai", "bangkok"],
    "mexico":    ["mexico", "mexican", "cdmx", "mexico city"],
    "generic":   ["manhole", "drain cover", "sewer cover"],  # fallback
}

# Minimum text records target per country
MIN_TEXT_PER_COUNTRY = 10

# ── Country Tagger ────────────────────────────────────────────────────────────

def infer_country(row):
    """Guess which country a post is about from its text content."""
    # Use existing country column if present and valid (from scraper)
    existing = row.get("country", "")
    if existing and existing != "unknown":
        return existing

    # Fallback: infer from text content
    text = f"{row.get('title','')} {row.get('text','')} {row.get('subreddit','')}".lower()

    for country, keywords in COUNTRY_KEYWORDS.items():
        if country == "generic":
            continue  # skip generic fallback for inference
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

def _pct_positive(s):
    """Named helper for aggregation — percentage of positive sentiment."""
    return (s == 1).mean() * 100

def _pct_negative(s):
    """Named helper for aggregation — percentage of negative sentiment."""
    return (s == -1).mean() * 100

def compute_weighted_sentiment(df):
    """Compute score-weighted mean sentiment per country (vectorized)."""

    if df.empty:
        return pd.DataFrame(columns=["weighted_sentiment", "n_posts",
                                     "mean_sentiment", "pct_positive", "pct_negative"])

    # Clip score to avoid outliers dominating (sqrt scale)
    df = df.copy()
    df["score_weight"] = df["score"].clip(lower=1) ** 0.5

    # Vectorized weighted average per country
    df["weighted_val"] = df["sentiment"] * df["score_weight"]
    grouped = df.groupby("country").agg(
        _wsum=("weighted_val", "sum"),
        _wden=("score_weight", "sum"),
        n_posts=("sentiment", "count"),
        mean_sentiment=("sentiment", "mean"),
        pct_positive=("sentiment", _pct_positive),
        pct_negative=("sentiment", _pct_negative),
    )

    # Compute weighted sentiment safely (avoid division by zero)
    grouped["weighted_sentiment"] = grouped["_wsum"] / grouped["_wden"].replace(0, 1)
    grouped = grouped.drop(columns=["_wsum", "_wden"])

    return grouped.sort_values("weighted_sentiment", ascending=False)


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

def plot_sentiment_by_country(summary_df, output_dir):
    """Figure 1: Weighted Sentiment Score by Country"""
    if summary_df.empty:
        print("  ⚠ Skipping sentiment_by_country: no data")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by weighted sentiment
    data = summary_df["weighted_sentiment"].sort_values(ascending=True)
    
    # Color based on polarity
    colors = ["#2ecc71" if v > 0.1 else "#e74c3c" if v < -0.1 else "#f39c12" for v in data]
    
    bars = ax.barh(data.index, data.values, color=colors, edgecolor="black", linewidth=0.5)
    
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Figure 1: Weighted Sentiment Score by Country\n(Positive = Green, Negative = Red, Neutral = Orange)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Weighted Sentiment Score (−1 to +1)", fontsize=11)
    ax.set_ylabel("Country", fontsize=11)
    
    # Add value labels
    for bar, val in zip(bars, data.values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.2f}", 
                va="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "sentiment_by_country.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: sentiment_by_country.png")


def plot_sentiment_composition(df, output_dir):
    """Figure 2: Sentiment Composition by Country (100% stacked bar)"""
    if df.empty:
        print("  ⚠ Skipping sentiment_composition: no data")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Calculate composition per country
    composition = df.groupby("country")["sentiment"].value_counts(normalize=True).unstack(fill_value=0)
    
    # Rename columns for clarity
    composition.columns = ["Negative" if c == -1 else "Neutral" if c == 0 else "Positive" for c in composition.columns]
    
    # Ensure all columns exist
    for col in ["Negative", "Neutral", "Positive"]:
        if col not in composition.columns:
            composition[col] = 0
    
    composition = composition[["Negative", "Neutral", "Positive"]]
    composition = composition.sort_values("Positive", ascending=False)
    
    # Create stacked bar
    x = range(len(composition))
    width = 0.7
    
    bars1 = ax.bar(x, composition["Negative"] * 100, width, label="Negative", color="#e74c3c")
    bars2 = ax.bar(x, composition["Neutral"] * 100, width, bottom=composition["Negative"] * 100, 
                   label="Neutral", color="#95a5a6")
    bars3 = ax.bar(x, composition["Positive"] * 100, width, 
                   bottom=(composition["Negative"] + composition["Neutral"]) * 100, 
                   label="Positive", color="#2ecc71")
    
    ax.set_xticks(x)
    ax.set_xticklabels(composition.index, rotation=45, ha="right")
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title("Figure 2: Sentiment Composition by Country\n(100% Stacked Bar Chart)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_dir / "sentiment_composition.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: sentiment_composition.png")


def plot_text_volume_by_country(df, output_dir, min_target=MIN_TEXT_PER_COUNTRY):
    """Figure 3: Text Samples Collected by Country"""
    if df.empty:
        print("  ⚠ Skipping text_volume_by_country: no data")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    
    counts = df.groupby("country").size().sort_values(ascending=False)
    
    bars = ax.bar(counts.index, counts.values, color="#3498db", edgecolor="black", linewidth=0.5)
    
    # Add minimum target line
    ax.axhline(min_target, color="#e74c3c", linewidth=2, linestyle="--", label=f"Target: {min_target}")
    
    ax.set_xlabel("Country", fontsize=11)
    ax.set_ylabel("Number of Text Records", fontsize=11)
    ax.set_title("Figure 3: Text Samples Collected by Country\n(Red line = minimum target)", fontsize=13, fontweight="bold")
    ax.legend()
    
    plt.xticks(rotation=45, ha="right")
    
    # Add value labels on bars
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, str(int(val)), 
                ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "text_volume_by_country.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: text_volume_by_country.png")


def plot_keyword_heatmap(kw_df, output_dir):
    """Figure 4: Design Vocabulary Distribution Heatmap"""
    if kw_df.empty:
        print("  ⚠ Skipping keyword_heatmap: no data")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Normalize per row so countries with more text don't dominate
    row_sums = kw_df.sum(axis=1).replace(0, 1)
    kw_norm = kw_df.div(row_sums, axis=0)
    
    sns.heatmap(kw_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=ax)
    ax.set_title("Figure 4: Design Vocabulary Distribution by Country\n(Normalized by row)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Vocabulary Category", fontsize=11)
    ax.set_ylabel("Country", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / "keyword_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: keyword_heatmap.png")


def plot_sentiment_confidence_distribution(df, output_dir):
    """Figure 5: Sentiment Confidence Distribution"""
    if df.empty:
        print("  ⚠ Skipping confidence_distribution: no data")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    
    countries = df["country"].unique()
    
    # Box plot of confidence by country
    data_to_plot = [df[df["country"] == c]["confidence"].values for c in countries if len(df[df["country"] == c]) > 0]
    valid_countries = [c for c in countries if len(df[df["country"] == c]) > 0]
    
    bp = ax.boxplot(data_to_plot, labels=valid_countries, patch_artist=True)
    
    # Color boxes
    colors = plt.cm.Set3(range(len(valid_countries)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel("Country", fontsize=11)
    ax.set_ylabel("Model Confidence Score", fontsize=11)
    ax.set_title("Figure 5: Sentiment Classification Confidence by Country", fontsize=13, fontweight="bold")
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: confidence_distribution.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load scraped data
    if not TEXT_CSV.exists():
        print(f"ERROR: {TEXT_CSV} not found. Run 01_scrape_data.py first.")
        return

    df = pd.read_csv(TEXT_CSV)
    print(f"Loaded {len(df)} text records")

    if df.empty:
        print("ERROR: No text records found. Run 01_scrape_data.py first.")
        return

    # Tag countries — prefer existing country column from scraper, infer only for unknowns
    if "country" in df.columns:
        # Re-infer only for rows where country is missing or "unknown"
        mask = df["country"].isna() | (df["country"] == "unknown") | (df["country"] == "")
        if mask.any():
            print(f"Inferring country for {mask.sum()} records without country tags...")
            df.loc[mask, "country"] = df.loc[mask].apply(infer_country, axis=1)
        else:
            print(f"All {len(df)} records already have country tags from scraper.")
    else:
        print("No country column found — inferring from text content...")
        df["country"] = df.apply(infer_country, axis=1)

    print(f"\nCountry distribution:\n{df['country'].value_counts().to_string()}\n")

    # Drop unknowns (can't compare them)
    n_unknown = (df["country"] == "unknown").sum()
    df = df[df["country"] != "unknown"]
    print(f"After removing {n_unknown} unknowns: {len(df)} records")

    if df.empty:
        print("ERROR: No records left after removing unknowns. Check data quality.")
        return

    # Run sentiment
    sentiment_pipe = load_sentiment_model()
    print(f"\nScoring sentiment on {len(df)} records...")
    scores_df = score_texts(df, sentiment_pipe)

    # Safety check: lengths must match before concat
    if len(df) != len(scores_df):
        print(f"⚠ Length mismatch: df={len(df)}, scores={len(scores_df)}. Trimming to match.")
        min_len = min(len(df), len(scores_df))
        df = df.iloc[:min_len].reset_index(drop=True)
        scores_df = scores_df.iloc[:min_len].reset_index(drop=True)

    df = pd.concat([df.reset_index(drop=True), scores_df], axis=1)

    # Save intermediate
    df.to_csv(SENTIMENT_CSV, index=False)
    print(f"\n✓ Saved sentiment data to {SENTIMENT_CSV}")

    # Aggregate
    print("\n=== Sentiment Summary ===")
    summary = compute_weighted_sentiment(df)
    print(summary.to_string())
    summary.to_csv(OUTPUT_DIR / "sentiment_summary.csv")

    # Keyword analysis
    print("\n=== Design Vocabulary ===")
    kw_df = keyword_frequency_by_country(df)
    print(kw_df.to_string())
    kw_df.to_csv(OUTPUT_DIR / "keyword_frequency.csv")

    # ── Generate Visualizations ─────────────────────────────────────────────
    print("\n=== Generating Visualizations ===")
    
    plot_sentiment_by_country(summary, OUTPUT_DIR)
    plot_sentiment_composition(df, OUTPUT_DIR)
    plot_text_volume_by_country(df, OUTPUT_DIR)
    plot_keyword_heatmap(kw_df, OUTPUT_DIR)
    plot_sentiment_confidence_distribution(df, OUTPUT_DIR)

    print("\n" + "=" * 50)
    print("✓ Sentiment Analysis Complete")
    print("=" * 50)
    print(f"\nGenerated 5 visualization figures:")
    print("  • Figure 1: sentiment_by_country.png")
    print("  • Figure 2: sentiment_composition.png")
    print("  • Figure 3: text_volume_by_country.png")
    print("  • Figure 4: keyword_heatmap.png")
    print("  • Figure 5: confidence_distribution.png")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nNext: run 04_image_processing.py")


if __name__ == "__main__":
    run()