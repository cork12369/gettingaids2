"""
STAGE 4: Cross Analysis
Cross-analyze sentiment and image data from all pipeline stages.
Generates comprehensive visualizations combining both text and image insights.

Reads from:
  /data/text_with_sentiment.csv  (from Stage 2)
  /data/output/image_analysis.csv (from Stage 3)

Outputs:
  /data/output/cross_analysis.csv
  /data/output/analysis_summary.csv
  /data/output/design_weights.json          ← correlation model (weighted scoring function)
  /data/output/cross_analysis_visualizations/
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr

# Paths
DATA_ROOT       = Path(os.getenv("DATA_DIR", "/data"))
TEXT_CSV        = DATA_ROOT / "text_with_sentiment.csv"
SENTIMENT_CSV   = DATA_ROOT / "output" / "sentiment_summary.csv"
IMAGE_CSV       = DATA_ROOT / "output" / "image_analysis.csv"
OUTPUT_DIR      = DATA_ROOT / "output"
CROSS_DIR       = DATA_ROOT / "output" / "cross_analysis_visualizations"
OUTPUT_CSV      = OUTPUT_DIR / "cross_analysis.csv"
SUMMARY_CSV     = OUTPUT_DIR / "analysis_summary.csv"
WEIGHTS_JSON    = OUTPUT_DIR / "design_weights.json"


# ── Visualization Functions ───────────────────────────────────────────────────

def plot_text_vs_image_by_country(text_df, img_df, output_dir):
    """Figure 1: Text Count vs Image Count by Country (Grouped Bar Chart)"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Aggregate by country
    text_counts = text_df.groupby("country").size() if "country" in text_df.columns else pd.Series()
    img_counts = img_df.groupby("country").size() if "country" in img_df.columns else pd.Series()
    
    # Combine into dataframe
    all_countries = sorted(set(text_counts.index) | set(img_counts.index))
    data = pd.DataFrame({
        "country": all_countries,
        "Text Records": [text_counts.get(c, 0) for c in all_countries],
        "Images": [img_counts.get(c, 0) for c in all_countries]
    })
    
    x = np.arange(len(all_countries))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, data["Text Records"], width, label="Text Records", color="#3498db")
    bars2 = ax.bar(x + width/2, data["Images"], width, label="Images", color="#e74c3c")
    
    ax.set_xlabel("Country", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Figure 1: Text Count vs Image Count by Country\n(Balanced Coverage Comparison)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(all_countries, rotation=45, ha="right")
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height, str(int(height)), 
                    ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height, str(int(height)), 
                    ha="center", va="bottom", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / "text_vs_image_by_country.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: text_vs_image_by_country.png")


def plot_sentiment_vs_image_volume(text_df, img_df, output_dir):
    """Figure 2: Sentiment vs Image Volume (Scatter Plot)"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate metrics per country
    sentiment_by_country = text_df.groupby("country")["sentiment"].mean() if "country" in text_df.columns else pd.Series()
    img_counts = img_df.groupby("country").size() if "country" in img_df.columns else pd.Series()
    
    # Combine
    countries = sorted(set(sentiment_by_country.index) & set(img_counts.index))
    
    if not countries:
        print("  ⚠ Not enough data for scatter plot")
        plt.close()
        return
    
    x = [img_counts.get(c, 0) for c in countries]
    y = [sentiment_by_country.get(c, 0) for c in countries]
    
    # Color by sentiment
    colors = ["#2ecc71" if s > 0.1 else "#e74c3c" if s < -0.1 else "#f39c12" for s in y]
    
    scatter = ax.scatter(x, y, c=colors, s=150, alpha=0.7, edgecolors="black", linewidth=1)
    
    # Add country labels
    for i, country in enumerate(countries):
        ax.annotate(country, (x[i], y[i]), fontsize=9, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")
    
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Number of Images", fontsize=11)
    ax.set_ylabel("Average Sentiment Score", fontsize=11)
    ax.set_title("Figure 2: Sentiment vs Image Volume by Country\n(Identifying Design Engagement Opportunities)", fontsize=13, fontweight="bold")
    
    # Add quadrant labels
    ax.text(0.95, 0.95, "High Visibility\nPositive Sentiment", transform=ax.transAxes, 
            fontsize=9, ha="right", va="top", style="italic", color="green")
    ax.text(0.95, 0.05, "High Visibility\nMixed Sentiment", transform=ax.transAxes, 
            fontsize=9, ha="right", va="bottom", style="italic", color="orange")
    ax.text(0.05, 0.95, "Underexplored\nPositive Potential", transform=ax.transAxes, 
            fontsize=9, ha="left", va="top", style="italic", color="blue")
    
    plt.tight_layout()
    plt.savefig(output_dir / "sentiment_vs_image_volume.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: sentiment_vs_image_volume.png")


def plot_combined_country_summary(text_df, img_df, sentiment_summary, output_dir):
    """Figure 3: Combined Cross-Country Summary (Combo Chart)"""
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Get data
    text_counts = text_df.groupby("country").size() if "country" in text_df.columns else pd.Series()
    img_counts = img_df.groupby("country").size() if "country" in img_df.columns else pd.Series()
    
    countries = sorted(set(text_counts.index) | set(img_counts.index))
    
    x = np.arange(len(countries))
    width = 0.35
    
    # Text counts (bars)
    bars1 = ax1.bar(x - width/2, [text_counts.get(c, 0) for c in countries], width, 
                    label="Text Records", color="#3498db", alpha=0.8)
    bars2 = ax1.bar(x + width/2, [img_counts.get(c, 0) for c in countries], width, 
                    label="Images", color="#e74c3c", alpha=0.8)
    
    ax1.set_xlabel("Country", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11, color="#333")
    ax1.tick_params(axis="y", labelcolor="#333")
    ax1.set_xticks(x)
    ax1.set_xticklabels(countries, rotation=45, ha="right")
    
    # Sentiment line (secondary axis)
    if sentiment_summary is not None and len(sentiment_summary) > 0:
        ax2 = ax1.twinx()
        sentiment_values = [sentiment_summary.get("weighted_sentiment", {}).get(c, 0) for c in countries]
        line = ax2.plot(x, sentiment_values, "o-", color="#2ecc71", linewidth=2, markersize=8, label="Weighted Sentiment")
        ax2.set_ylabel("Weighted Sentiment", fontsize=11, color="#2ecc71")
        ax2.tick_params(axis="y", labelcolor="#2ecc71")
        ax2.axhline(0, color="#2ecc71", linewidth=0.5, linestyle="--", alpha=0.5)
    
    ax1.set_title("Figure 3: Combined Cross-Country Summary\n(Counts + Sentiment Overlay)", fontsize=13, fontweight="bold")
    
    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    if sentiment_summary is not None and len(sentiment_summary) > 0:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(output_dir / "combined_country_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: combined_country_summary.png")


def plot_balance_ratio_chart(text_df, img_df, output_dir):
    """Figure 4: Text-to-Image Ratio by Country"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate ratios
    text_counts = text_df.groupby("country").size() if "country" in text_df.columns else pd.Series()
    img_counts = img_df.groupby("country").size() if "country" in img_df.columns else pd.Series()
    
    all_countries = sorted(set(text_counts.index) | set(img_counts.index))
    ratios = []
    for c in all_countries:
        t = text_counts.get(c, 0)
        i = img_counts.get(c, 0)
        if i > 0:
            ratios.append(t / i)
        else:
            ratios.append(0)
    
    # Ideal ratio line
    ideal_ratio = 1.0
    
    colors = ["#2ecc71" if 0.8 <= r <= 1.5 else "#f39c12" if 0.5 <= r < 0.8 or r > 1.5 else "#e74c3c" for r in ratios]
    
    bars = ax.bar(all_countries, ratios, color=colors, edgecolor="black", linewidth=0.5)
    
    ax.axhline(ideal_ratio, color="#3498db", linewidth=2, linestyle="--", label="Ideal Balance (1:1)")
    
    ax.set_xlabel("Country", fontsize=11)
    ax.set_ylabel("Text-to-Image Ratio", fontsize=11)
    ax.set_title("Figure 4: Text-to-Image Ratio by Country\n(Green = Balanced, Orange = Slight Imbalance, Red = Imbalanced)", fontsize=13, fontweight="bold")
    ax.legend()
    
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(output_dir / "balance_ratio_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: balance_ratio_chart.png")


def plot_coverage_summary(text_df, img_df, output_dir):
    """Figure 5: Overall Coverage Summary (Pie Charts)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Text coverage pie
    text_counts = text_df.groupby("country").size() if "country" in text_df.columns else pd.Series()
    if len(text_counts) > 0:
        colors1 = plt.cm.Set3(range(len(text_counts)))
        axes[0].pie(text_counts, labels=text_counts.index, autopct="%1.1f%%", colors=colors1, startangle=90)
        axes[0].set_title("Text Records by Country", fontsize=11, fontweight="bold")
    else:
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center")
    
    # Image coverage pie
    img_counts = img_df.groupby("country").size() if "country" in img_df.columns else pd.Series()
    if len(img_counts) > 0:
        colors2 = plt.cm.Set3(range(len(img_counts)))
        axes[1].pie(img_counts, labels=img_counts.index, autopct="%1.1f%%", colors=colors2, startangle=90)
        axes[1].set_title("Images by Country", fontsize=11, fontweight="bold")
    else:
        axes[1].text(0.5, 0.5, "No data", ha="center", va="center")
    
    plt.suptitle("Figure 5: Overall Coverage Summary", fontsize=13, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_dir / "coverage_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: coverage_summary.png")


def plot_human_ai_agreement(text_df, output_dir):
    """Figure 7: Human vs AI Sentiment Agreement by Country."""
    human_grades_path = DATA_ROOT / "human_grades.csv"
    grade_sample_path = DATA_ROOT / "grade_sample.csv"
    if not human_grades_path.exists() or not grade_sample_path.exists():
        print("  ⚠ No human grades yet — skipping Human vs AI agreement chart")
        return

    import pandas as pd
    grades = pd.read_csv(human_grades_path)
    sample = pd.read_csv(grade_sample_path)

    # Merge grades with sample to get country and AI label
    sample["snippet_id"] = sample["snippet_id"].astype(str)
    grades["snippet_id"] = grades["snippet_id"].astype(str)
    merged = grades.merge(sample[["snippet_id", "country", "label"]], on="snippet_id", how="left")

    # Map scores to labels: human_score 0→NEG, 1→NEU, 2→POS
    def _human_label(s):
        s = int(s)
        return {-1: "NEG", 0: "NEU", 1: "POS"}.get(s, "NEU")
    # Scores are 0=neg, 1=neu, 2=pos in the grading UI
    merged["human_label"] = merged["human_score"].map({0: "NEG", 1: "NEU", 2: "POS"})
    merged["ai_short"] = merged["label"].str.upper().str.replace("ITIVE", "").str.replace("GATIVE", "").str.strip()
    merged["ai_short"] = merged["label"].apply(
        lambda x: "POS" if "POS" in str(x).upper() else ("NEG" if "NEG" in str(x).upper() else "NEU"))
    merged["agree"] = merged["human_label"] == merged["ai_short"]

    if "country" not in merged.columns or merged["country"].dropna().empty:
        print("  ⚠ No country data in human grades — skipping agreement chart")
        return

    agreement = merged.groupby("country")["agree"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2ecc71" if v >= 0.7 else "#f39c12" if v >= 0.5 else "#e74c3c" for v in agreement.values]
    bars = ax.bar(agreement.index, agreement.values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, agreement.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 1.1)
    ax.axhline(0.7, color="#2ecc71", linewidth=1, linestyle="--", alpha=0.5, label="Good (70%)")
    ax.axhline(0.5, color="#f39c12", linewidth=1, linestyle="--", alpha=0.5, label="Fair (50%)")
    ax.set_xlabel("Country", fontsize=11)
    ax.set_ylabel("Agreement Rate", fontsize=11)
    ax.set_title(f"Figure 7: Human vs AI Sentiment Agreement by Country\n"
                 f"(Based on {len(merged)} human-graded snippets)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "human_ai_agreement.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: human_ai_agreement.png")


def plot_sentiment_heatmap(text_df, output_dir):
    """Figure 6: Sentiment Distribution Heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pivot table of sentiment counts by country
    sentiment_counts = text_df.groupby(["country", "sentiment"]).size().unstack(fill_value=0)
    sentiment_counts.columns = ["Negative" if c == -1 else "Neutral" if c == 0 else "Positive" for c in sentiment_counts.columns]
    sentiment_counts = sentiment_counts[["Negative", "Neutral", "Positive"]]
    
    # Normalize
    sentiment_norm = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0)
    
    sns.heatmap(sentiment_norm, annot=True, fmt=".2f", cmap="RdYlGn", linewidths=0.5, ax=ax)
    
    ax.set_title("Figure 6: Sentiment Distribution Heatmap\n(Normalized by Country)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sentiment Category", fontsize=11)
    ax.set_ylabel("Country", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / "sentiment_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: sentiment_heatmap.png")


# ── Design-Weight Correlation Model ───────────────────────────────────────────

# Ordinal encoding maps for VLM categorical attributes → 0-1 numeric scale
ORNAMENTATION_MAP = {
    "plain": 0.0, "minimal": 0.25, "moderate": 0.5,
    "ornate": 0.75, "highly_ornate": 1.0,
}
AESTHETIC_MAP = {
    "low": 0.0, "medium": 0.5, "high": 1.0,
}

# The four design attributes used in the correlation model
DESIGN_ATTRIBUTES = [
    "ornamentation_level",
    "cultural_elements",
    "aesthetic_appeal",
    "motif_diversity",
]


def _encode_image_attributes(img_df: pd.DataFrame) -> pd.DataFrame:
    """Encode VLM categorical attributes to numeric scores (0-1 scale).

    Returns a copy of img_df with new numeric columns:
      ornamentation_num, cultural_elements_num, aesthetic_appeal_num, motif_diversity_num
    """
    df = img_df.copy()

    # Ornamentation: ordinal string → 0-1
    if "ornamentation_level" in df.columns:
        df["ornamentation_num"] = (
            df["ornamentation_level"]
            .astype(str)
            .str.lower()
            .str.strip()
            .map(ORNAMENTATION_MAP)
            .fillna(0.25)  # default to "minimal" if unknown
        )
    else:
        df["ornamentation_num"] = 0.25

    # Cultural elements: boolean → 0 / 1
    if "cultural_elements" in df.columns:
        df["cultural_elements_num"] = df["cultural_elements"].apply(
            lambda x: 1.0 if str(x).lower() in ("true", "yes", "1") else 0.0
        )
    else:
        df["cultural_elements_num"] = 0.0

    # Aesthetic appeal: ordinal string → 0-1
    if "aesthetic_appeal" in df.columns:
        df["aesthetic_appeal_num"] = (
            df["aesthetic_appeal"]
            .astype(str)
            .str.lower()
            .str.strip()
            .map(AESTHETIC_MAP)
            .fillna(0.5)  # default to "medium"
        )
    else:
        df["aesthetic_appeal_num"] = 0.5

    # Motif diversity: count of unique motif types per image
    if "motifs" in df.columns:
        df["motif_diversity_num"] = df["motifs"].apply(
            lambda x: min(
                len([m for m in str(x).split("|") if m and m != "none"]) / 5.0,
                1.0,
            )
        )
    else:
        df["motif_diversity_num"] = 0.0

    return df


def compute_design_weights(text_df: pd.DataFrame, img_df: pd.DataFrame) -> dict:
    """Merge VLM attributes with sentiment scores by country, compute correlations,
    and produce a weighted scoring function saved as design_weights.json.

    Returns the weights dict (also written to WEIGHTS_JSON).
    """
    import json

    print("\n=== Computing Design–Sentiment Correlation Model ===")

    # ── 1. Encode image attributes ──────────────────────────────────────────
    img_encoded = _encode_image_attributes(img_df)

    # Filter to actual manhole covers if the field exists
    if "is_manhole_cover" in img_encoded.columns:
        before = len(img_encoded)
        img_encoded = img_encoded[
            img_encoded["is_manhole_cover"].apply(
                lambda x: str(x).lower() in ("true", "yes", "1")
            )
        ]
        print(f"  Filtered to manhole covers: {len(img_encoded)} / {before} images")

    # ── 2. Aggregate image attributes by country (mean) ─────────────────────
    num_cols = [c for c in img_encoded.columns if c.endswith("_num")]
    img_by_country = (
        img_encoded.groupby("country")[num_cols]
        .mean()
        .rename(columns={c: c.replace("_num", "") for c in num_cols})
    )

    # ── 3. Aggregate sentiment by country ───────────────────────────────────
    if "score" in text_df.columns and "sentiment" in text_df.columns:
        # Score-weighted sentiment (same logic as 03_sentiment_analysis.py)
        tdf = text_df.copy()
        tdf["score_weight"] = tdf["score"].clip(lower=1) ** 0.5
        tdf["weighted_val"] = tdf["sentiment"] * tdf["score_weight"]
        sent = tdf.groupby("country").agg(
            _wsum=("weighted_val", "sum"), _wden=("score_weight", "sum")
        )
        sent_by_country = (sent["_wsum"] / sent["_wden"].replace(0, 1)).rename(
            "sentiment"
        )
    else:
        sent_by_country = (
            text_df.groupby("country")["sentiment"].mean().rename("sentiment")
        )

    # ── 4. Merge on country ─────────────────────────────────────────────────
    merged = img_by_country.join(sent_by_country, how="inner").dropna()

    if len(merged) < 2:
        print("  ⚠ Fewer than 2 countries with both image & sentiment data — "
              "cannot compute correlations. Using equal weights.")
        weights = {attr: round(1.0 / len(DESIGN_ATTRIBUTES), 4)
                   for attr in DESIGN_ATTRIBUTES}
        correlations = {attr: 0.0 for attr in DESIGN_ATTRIBUTES}
        country_benchmarks = {}
    else:
        print(f"  Computing correlations across {len(merged)} countries: "
              f"{sorted(merged.index.tolist())}")

        # ── 5. Pearson correlation per attribute vs sentiment ────────────────
        correlations = {}
        for attr in DESIGN_ATTRIBUTES:
            if attr in merged.columns:
                r = merged[attr].corr(merged["sentiment"])
                correlations[attr] = round(r, 4)
            else:
                correlations[attr] = 0.0

        print("  Raw correlations:")
        for attr, r in correlations.items():
            print(f"    {attr:25s}  r = {r:+.4f}")

        # ── 6. Convert correlations → weights (abs, normalised to sum=1) ────
        abs_corrs = {k: abs(v) for k, v in correlations.items()}
        total = sum(abs_corrs.values())

        if total == 0:
            # All correlations zero — fall back to equal weights
            weights = {attr: round(1.0 / len(DESIGN_ATTRIBUTES), 4)
                       for attr in DESIGN_ATTRIBUTES}
        else:
            weights = {
                attr: round(abs_corrs[attr] / total, 4)
                for attr in DESIGN_ATTRIBUTES
            }

        # ── 7. Country benchmarks ───────────────────────────────────────────
        country_benchmarks = {}
        for country in merged.index:
            row = merged.loc[country]
            benchmark = {"sentiment": round(float(row["sentiment"]), 4)}
            for attr in DESIGN_ATTRIBUTES:
                if attr in row.index:
                    benchmark[attr] = round(float(row[attr]), 4)
            country_benchmarks[country] = benchmark

    # ── 8. Build output dict ────────────────────────────────────────────────
    result = {
        "weights": weights,
        "correlations": correlations,
        "country_benchmarks": country_benchmarks,
        "n_countries": len(merged) if len(merged) >= 2 else 0,
        "method": "pearson_correlation_normalized",
        "generated_at": datetime.now().isoformat(),
        "description": (
            "Weights derived from Pearson correlation between VLM visual "
            "attributes and public sentiment, aggregated by country. "
            "Use: predicted_sentiment = sum(score[attr] * weight "
            "for attr, weight in weights.items())"
        ),
    }

    # ── 9. Save JSON ────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_JSON.write_text(json.dumps(result, indent=2))
    print(f"\n  ✓ Saved design weights → {WEIGHTS_JSON}")

    # ── 10. Print summary ───────────────────────────────────────────────────
    print("\n  ── Design Weights (weighted scoring function) ──")
    for attr, w in weights.items():
        r = correlations[attr]
        print(f"    {attr:25s}  weight = {w:.4f}  (r = {r:+.4f})")
    print(f"    {'TOTAL':25s}  weight = {sum(weights.values()):.4f}")

    if country_benchmarks:
        print("\n  ── Country Benchmarks ──")
        for country, bm in sorted(country_benchmarks.items()):
            pred = sum(
                bm.get(attr, 0) * weights[attr] for attr in DESIGN_ATTRIBUTES
            )
            print(f"    {country:15s}  sentiment={bm['sentiment']:+.3f}  "
                  f"predicted={pred:+.3f}")

    return result


def plot_correlation_heatmap(weights_data: dict, output_dir: Path):
    """Figure 8: Design Attribute ↔ Sentiment Correlation Heatmap."""
    benchmarks = weights_data.get("country_benchmarks", {})
    if not benchmarks:
        print("  ⚠ No country benchmarks — skipping correlation heatmap")
        return

    # Build DataFrame: countries × design attributes + sentiment
    rows = []
    for country, bm in benchmarks.items():
        row = {"country": country}
        row.update(bm)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("country")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 1]})

    # Left: heatmap of attributes × countries
    attr_cols = [c for c in df.columns]  # sentiment + design attributes
    sns.heatmap(
        df[attr_cols],
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        ax=axes[0],
    )
    axes[0].set_title(
        "Figure 8a: Design Attributes & Sentiment by Country",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_ylabel("Country")
    axes[0].set_xlabel("Attribute")

    # Right: correlation bar chart
    correlations = weights_data.get("correlations", {})
    weights = weights_data.get("weights", {})
    if correlations:
        attrs = list(correlations.keys())
        r_vals = [correlations[a] for a in attrs]
        w_vals = [weights.get(a, 0) for a in attrs]

        x = np.arange(len(attrs))
        width = 0.35
        bars_r = axes[1].barh(
            x - width / 2, r_vals, width, label="Correlation (r)",
            color=["#2ecc71" if v >= 0 else "#e74c3c" for v in r_vals],
        )
        bars_w = axes[1].barh(
            x + width / 2, w_vals, width, label="Weight",
            color="#3498db", alpha=0.7,
        )
        axes[1].set_yticks(x)
        axes[1].set_yticklabels(attrs)
        axes[1].axvline(0, color="gray", linewidth=0.5, linestyle="--")
        axes[1].set_title("Figure 8b: Correlations & Weights",
                          fontsize=12, fontweight="bold")
        axes[1].legend(fontsize=8)
        axes[1].set_xlabel("Value")
    else:
        axes[1].text(0.5, 0.5, "No correlations", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: correlation_heatmap.png")


# ── Vocabulary–Visual Cross-Correlation ───────────────────────────────────────

# Design vocabulary categories from text analytics (03_sentiment_analysis.py)
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


def _vocab_frequency_by_country(text_df: pd.DataFrame) -> pd.DataFrame:
    """Compute normalised design-vocabulary frequency per country."""
    records = []
    for country, group in text_df.groupby("country"):
        text_blob = " ".join(group["text"].fillna("").str.lower())
        row = {"country": country}
        for category, words in DESIGN_VOCAB.items():
            row[category] = sum(text_blob.count(w) for w in words)
        records.append(row)
    kw_df = pd.DataFrame(records).set_index("country")
    # Row-normalise so countries with more text don't dominate
    row_sums = kw_df.sum(axis=1).replace(0, 1)
    return kw_df.div(row_sums, axis=0)


def compute_vocab_visual_correlation(text_df: pd.DataFrame,
                                      img_df: pd.DataFrame) -> dict:
    """Cross-correlate DESIGN_VOCAB categories with VLM visual attributes.

    Produces:
      - correlation_matrix: Pearson r for each (vocab_category, visual_attribute) pair
      - p_values: statistical significance of each pair
      - significant_pairs: pairs where p < 0.05
      - hypothesis_tests: targeted tests for specific style→vocab hypotheses
    """
    print("\n=== Computing Vocabulary × Visual Attribute Correlation ===")

    # ── 1. Vocab frequency per country (normalised) ─────────────────────────
    vocab_df = _vocab_frequency_by_country(text_df)
    print(f"  Vocab profiles: {len(vocab_df)} countries")

    # ── 2. Encode and aggregate visual attributes by country ────────────────
    img_encoded = _encode_image_attributes(img_df)

    # Filter to manhole covers
    if "is_manhole_cover" in img_encoded.columns:
        img_encoded = img_encoded[
            img_encoded["is_manhole_cover"].apply(
                lambda x: str(x).lower() in ("true", "yes", "1")
            )
        ]

    num_cols = [c for c in img_encoded.columns if c.endswith("_num")]
    visual_by_country = (
        img_encoded.groupby("country")[num_cols]
        .mean()
        .rename(columns={c: c.replace("_num", "") for c in num_cols})
    )

    # Also aggregate dominant_style mode per country
    if "dominant_style" in img_encoded.columns:
        style_mode = (
            img_encoded.groupby("country")["dominant_style"]
            .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else "unknown")
            .rename("dominant_style")
        )
        visual_by_country = visual_by_country.join(style_mode, how="left")

    # Also get mean ornamentation_level by country for hypothesis testing
    if "ornamentation_level" in img_encoded.columns:
        orn_mean = (
            img_encoded.groupby("country")["ornamentation_num"]
            .mean()
            .rename("ornamentation_mean")
        )
        visual_by_country = visual_by_country.join(orn_mean, how="left")

    print(f"  Visual profiles: {len(visual_by_country)} countries")

    # ── 3. Merge on country ─────────────────────────────────────────────────
    merged = vocab_df.join(visual_by_country, how="inner")

    # Drop non-numeric columns for correlation
    numeric_cols = [c for c in merged.columns if merged[c].dtype in (np.float64, np.int64, float, int)]

    if len(merged) < 3:
        print("  ⚠ Fewer than 3 countries with both vocab & visual data — "
              "correlations unreliable. Reporting raw data only.")
        return {
            "correlation_matrix": {},
            "p_values": {},
            "significant_pairs": [],
            "hypothesis_tests": {},
            "n_countries": len(merged),
            "countries": sorted(merged.index.tolist()),
        }

    print(f"  Correlating across {len(merged)} countries: "
          f"{sorted(merged.index.tolist())}")

    # ── 4. Pearson correlation matrix: vocab × visual ───────────────────────
    vocab_categories = list(DESIGN_VOCAB.keys())
    visual_attributes = [a for a in DESIGN_ATTRIBUTES if a in numeric_cols]

    correlation_matrix = {}
    p_values = {}
    significant_pairs = []

    for vcat in vocab_categories:
        correlation_matrix[vcat] = {}
        p_values[vcat] = {}
        for vattr in visual_attributes:
            if vcat in merged.columns and vattr in merged.columns:
                x = merged[vcat].values
                y = merged[vattr].values
                if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
                    r, p = pearsonr(x, y)
                    correlation_matrix[vcat][vattr] = round(r, 4)
                    p_values[vcat][vattr] = round(p, 6)
                    if p < 0.05:
                        stars = "**" if p < 0.01 else "*"
                        significant_pairs.append({
                            "vocab": vcat,
                            "visual": vattr,
                            "r": round(r, 4),
                            "p": round(p, 6),
                            "stars": stars,
                        })
                else:
                    correlation_matrix[vcat][vattr] = 0.0
                    p_values[vcat][vattr] = 1.0
            else:
                correlation_matrix[vcat][vattr] = 0.0
                p_values[vcat][vattr] = 1.0

    # ── 5. Print correlation matrix ─────────────────────────────────────────
    print("\n  ── Vocabulary × Visual Correlation Matrix ──")
    header = f"  {'':20s}" + "".join(f"{a:>18s}" for a in visual_attributes)
    print(header)
    for vcat in vocab_categories:
        row_str = f"  {vcat:20s}"
        for vattr in visual_attributes:
            r = correlation_matrix[vcat].get(vattr, 0.0)
            p = p_values[vcat].get(vattr, 1.0)
            stars = "**" if p < 0.01 else "*" if p < 0.05 else ""
            row_str += f"  {r:+.3f}{stars:>4s}"
        print(row_str)

    if significant_pairs:
        print(f"\n  Significant pairs (p < 0.05): {len(significant_pairs)}")
        for sp in significant_pairs:
            print(f"    {sp['vocab']:12s} ↔ {sp['visual']:25s}  "
                  f"r={sp['r']:+.4f}  p={sp['p']:.4f} {sp['stars']}")
    else:
        print("\n  No significant pairs at p < 0.05 level")

    # ── 6. Hypothesis tests ─────────────────────────────────────────────────
    hypothesis_tests = {}

    # Hypothesis 1: "industrial" countries → higher "functional" vocab
    if "dominant_style" in merged.columns:
        industrial_countries = merged[
            merged["dominant_style"] == "industrial"
        ].index.tolist()
        other_countries = merged[
            merged["dominant_style"] != "industrial"
        ].index.tolist()

        if industrial_countries and other_countries:
            ind_func = merged.loc[industrial_countries, "functional"].mean()
            oth_func = merged.loc[other_countries, "functional"].mean()
            hypothesis_tests["industrial_functional"] = {
                "hypothesis": "Countries with 'industrial' dominant style "
                              "use more 'functional' vocabulary",
                "countries": industrial_countries,
                "mean_functional_score": round(ind_func, 4),
                "mean_functional_other": round(oth_func, 4),
                "difference": round(ind_func - oth_func, 4),
                "supports_hypothesis": ind_func > oth_func,
            }
            print(f"\n  ── Hypothesis: Industrial style → Functional vocab ──")
            print(f"    Industrial countries: {industrial_countries}")
            print(f"    Functional vocab (industrial): {ind_func:.4f}")
            print(f"    Functional vocab (others):     {oth_func:.4f}")
            print(f"    Supports hypothesis: {'YES' if ind_func > oth_func else 'NO'}")

    # Hypothesis 2: "highly_ornate" countries → higher "engagement" vocab
    if "ornamentation_mean" in merged.columns:
        # Top quartile ornamentation = "highly ornate" countries
        orn_threshold = merged["ornamentation_mean"].quantile(0.75)
        ornate_countries = merged[
            merged["ornamentation_mean"] >= orn_threshold
        ].index.tolist()
        plain_countries = merged[
            merged["ornamentation_mean"] < orn_threshold
        ].index.tolist()

        if ornate_countries and plain_countries:
            orn_engage = merged.loc[ornate_countries, "engagement"].mean()
            pln_engage = merged.loc[plain_countries, "engagement"].mean()
            hypothesis_tests["ornate_engagement"] = {
                "hypothesis": "Countries with higher ornamentation "
                              "use more 'engagement' vocabulary",
                "countries": ornate_countries,
                "ornamentation_threshold": round(orn_threshold, 4),
                "mean_engagement_score": round(orn_engage, 4),
                "mean_engagement_other": round(pln_engage, 4),
                "difference": round(orn_engage - pln_engage, 4),
                "supports_hypothesis": orn_engage > pln_engage,
            }
            print(f"\n  ── Hypothesis: Highly ornate → Engagement vocab ──")
            print(f"    Ornate countries (top quartile): {ornate_countries}")
            print(f"    Engagement vocab (ornate):  {orn_engage:.4f}")
            print(f"    Engagement vocab (others):  {pln_engage:.4f}")
            print(f"    Supports hypothesis: {'YES' if orn_engage > pln_engage else 'NO'}")

    # Hypothesis 3: "aesthetic" vocab correlates with "aesthetic_appeal" visual
    if "aesthetic" in merged.columns and "aesthetic_appeal" in merged.columns:
        r_ae, p_ae = pearsonr(merged["aesthetic"].values,
                               merged["aesthetic_appeal"].values)
        hypothesis_tests["aesthetic_vocab_aesthetic_visual"] = {
            "hypothesis": "Countries with more 'aesthetic' vocabulary "
                          "also have higher VLM aesthetic_appeal scores",
            "r": round(r_ae, 4),
            "p": round(p_ae, 6),
            "significant": p_ae < 0.05,
            "supports_hypothesis": r_ae > 0,
        }
        print(f"\n  ── Hypothesis: Aesthetic vocab ↔ Aesthetic visual ──")
        print(f"    r = {r_ae:+.4f}, p = {p_ae:.4f}")
        print(f"    Supports hypothesis: {'YES' if r_ae > 0 else 'NO'}")

    # ── 7. Build and save result ────────────────────────────────────────────
    result = {
        "correlation_matrix": correlation_matrix,
        "p_values": p_values,
        "significant_pairs": significant_pairs,
        "hypothesis_tests": hypothesis_tests,
        "n_countries": len(merged),
        "countries": sorted(merged.index.tolist()),
        "vocab_categories": vocab_categories,
        "visual_attributes": visual_attributes,
        "method": "pearson_correlation_vocab_visual",
        "generated_at": datetime.now().isoformat(),
    }

    vocab_json_path = OUTPUT_DIR / "vocab_visual_correlation.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vocab_json_path.write_text(json.dumps(result, indent=2))
    print(f"\n  ✓ Saved vocab-visual correlation → {vocab_json_path}")

    return result


def plot_vocab_visual_heatmap(corr_data: dict, output_dir: Path):
    """Figure 9: Vocabulary × Visual Attribute Correlation Heatmap.

    Rows = vocab categories, columns = visual attributes.
    Cells show Pearson r with significance stars.
    """
    corr_matrix = corr_data.get("correlation_matrix", {})
    p_val_matrix = corr_data.get("p_values", {})
    vocab_cats = corr_data.get("vocab_categories", list(DESIGN_VOCAB.keys()))
    visual_attrs = corr_data.get("visual_attributes", DESIGN_ATTRIBUTES)

    if not corr_matrix:
        print("  ⚠ No vocab-visual correlation data — skipping heatmap")
        return

    # Build DataFrames
    r_df = pd.DataFrame(corr_matrix).T
    r_df = r_df.reindex(index=vocab_cats, columns=visual_attrs)

    p_df = pd.DataFrame(p_val_matrix).T
    p_df = p_df.reindex(index=vocab_cats, columns=visual_attrs)

    # Build annotation strings: r value + significance stars
    annot = r_df.copy().astype(str)
    for row in annot.index:
        for col in annot.columns:
            r_val = r_df.loc[row, col]
            p_val = p_df.loc[row, col]
            try:
                r_f = float(r_val)
                p_f = float(p_val)
            except (ValueError, TypeError):
                annot.loc[row, col] = ""
                continue
            stars = "**" if p_f < 0.01 else "*" if p_f < 0.05 else ""
            annot.loc[row, col] = f"{r_f:+.2f}{stars}"

    fig, ax = plt.subplots(figsize=(12, 7))

    # Convert to float for colormap
    heat_data = r_df.astype(float)

    sns.heatmap(
        heat_data,
        annot=annot,
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.8,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Pearson r"},
    )

    ax.set_title(
        "Figure 9: Text Vocabulary × Visual Attribute Correlation\n"
        "(Pearson r by country, * p<0.05, ** p<0.01)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("VLM Visual Attribute", fontsize=11)
    ax.set_ylabel("Text Vocabulary Category", fontsize=11)

    # Rotate x labels for readability
    ax.set_xticklabels(
        [l.get_text().replace("_", " ").title() for l in ax.get_xticklabels()],
        rotation=30,
        ha="right",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "vocab_visual_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: vocab_visual_heatmap.png")


# ── Design Requirements Generator ──────────────────────────────────────────────

# Human-readable labels for each VLM attribute
ATTR_LABELS = {
    "ornamentation_level": "ornamentation level",
    "cultural_elements":   "cultural element integration",
    "aesthetic_appeal":    "aesthetic appeal",
    "motif_diversity":     "motif diversity",
}

# Requirement templates keyed by (category, direction_sign)
VISUAL_REQ_TEMPLATES = {
    (+1, "ornamentation_level"):  "Increase ornamentation complexity to improve public sentiment — ornate designs correlate positively with approval.",
    (-1, "ornamentation_level"):  "Reduce ornamentation complexity — simpler designs correlate with higher public approval in the sampled countries.",
    (+1, "cultural_elements"):    "Integrate localized cultural motifs to maximize public engagement — cultural elements strongly boost sentiment.",
    (-1, "cultural_elements"):    "Minimize overt cultural motifs — sentiment data suggests universal/neutral designs perform better.",
    (+1, "aesthetic_appeal"):     "Prioritize high aesthetic appeal in cover design — visually attractive covers receive significantly more positive sentiment.",
    (-1, "aesthetic_appeal"):     "Aesthetic appeal shows limited or negative correlation with sentiment — focus engineering quality over decoration.",
    (+1, "motif_diversity"):      "Incorporate diverse visual motifs — variety in design elements correlates with higher public engagement.",
    (-1, "motif_diversity"):      "Limit motif diversity — focused, cohesive designs outcover busy compositions in sentiment ratings.",
}

VOCAB_REQ_TEMPLATES = {
    ("aesthetic",  "ornamentation_level", +1): "Aesthetic vocabulary correlates with higher ornamentation — leverage ornate designs as tourist/collection attractions.",
    ("aesthetic",  "ornamentation_level", -1): "Aesthetic vocabulary correlates with simpler designs — refine minimal aesthetics for broader appeal.",
    ("functional", "ornamentation_level", -1): "Functional vocabulary associates with plain covers — maintain industrial simplicity for safety-critical installations.",
    ("functional", "ornamentation_level", +1): "Functional vocabulary surprisingly associates with ornate covers — ensure decorative elements do not compromise perceived safety.",
    ("cultural",   "cultural_elements",   +1): "Cultural vocabulary aligns with cultural visual elements — strengthen local identity motifs for community engagement.",
    ("engagement", "ornamentation_level", +1): "Engagement vocabulary (collect, photograph, tourist) strongly correlates with ornate covers — design for 'collectibility' in cultural districts.",
    ("engagement", "ornamentation_level", -1): "Engagement vocabulary correlates with simpler designs — consider that minimalism can also drive public interest.",
    ("negative",   "aesthetic_appeal",    -1): "Negative vocabulary inversely correlates with aesthetic appeal — poor aesthetics trigger negative public reactions; invest in visual quality.",
    ("negative",   "ornamentation_level", +1): "Negative vocabulary correlates with higher ornamentation — avoid over-decoration that may be perceived as cluttered or excessive.",
}

HYPOTHESIS_REQ_TEMPLATES = {
    "industrial_functional": {
        True:  "Hypothesis confirmed: Industrial-style covers (Germany/UK) are perceived as functional — maintain this design language for utilitarian/safety contexts.",
        False: "Hypothesis not confirmed: Industrial style does not associate with functional vocabulary — reconsider design language assumptions.",
    },
    "ornate_engagement": {
        True:  "Hypothesis confirmed: Highly ornate covers drive tourism and collection behavior — leverage ornate designs for cultural/tourism districts.",
        False: "Hypothesis not confirmed: Ornamentation does not drive engagement vocabulary — focus on other engagement triggers.",
    },
    "aesthetic_vocab_aesthetic_visual": {
        True:  "Hypothesis confirmed: Public aesthetic vocabulary aligns with VLM aesthetic scores — VLM scoring is a valid proxy for human aesthetic judgment.",
        False: "Hypothesis not confirmed: VLM aesthetic scores diverge from public aesthetic vocabulary — calibrate VLM scoring with human feedback.",
    },
}


def generate_design_requirements(weights_data: dict | None,
                                  vocab_corr: dict | None) -> pd.DataFrame:
    """Auto-generate design requirements from correlation evidence.

    Uses the actual computed correlations to produce evidence-backed design
    requirements — no manual guesswork.  Outputs design_requirements.csv.

    Returns the requirements DataFrame.
    """
    print("\n=== Generating Evidence-Based Design Requirements ===")

    requirements = []
    dr_id = 0

    def _add(cat, req, evidence, r_val, direction, source):
        nonlocal dr_id
        dr_id += 1
        requirements.append({
            "id":         f"DR-{dr_id:02d}",
            "category":   cat,
            "requirement": req,
            "evidence":   evidence,
            "correlation_r": r_val,
            "direction":  direction,
            "source":     source,
        })

    # ── 1. Visual → Sentiment requirements (from design weights) ──────────
    if weights_data and weights_data.get("correlations"):
        correlations = weights_data["correlations"]
        for attr in DESIGN_ATTRIBUTES:
            r = correlations.get(attr, 0.0)
            sign = +1 if r >= 0 else -1

            # Only emit requirement if |r| exceeds threshold
            if abs(r) >= 0.15:
                template = VISUAL_REQ_TEMPLATES.get((sign, attr))
                if template:
                    evidence = (f"{attr} vs sentiment r={r:+.4f}, "
                                f"weight={weights_data['weights'].get(attr, 0):.4f}")
                    direction = "positive" if r > 0 else "negative"
                    _add("visual_sentiment", template, evidence,
                         round(r, 4), direction, "design_weights")

    # ── 2. Vocab × Visual requirements (from vocab-visual correlation) ────
    if vocab_corr and vocab_corr.get("correlation_matrix"):
        corr_matrix = vocab_corr["correlation_matrix"]
        p_matrix    = vocab_corr.get("p_values", {})

        for (vcat, vattr, sign), template in VOCAB_REQ_TEMPLATES.items():
            r = corr_matrix.get(vcat, {}).get(vattr, 0.0)
            p = p_matrix.get(vcat, {}).get(vattr, 1.0)
            # Only emit if correlation direction matches template AND |r| >= 0.15
            actual_sign = +1 if r >= 0 else -1
            if actual_sign == sign and abs(r) >= 0.15:
                stars = "**" if p < 0.01 else "*" if p < 0.05 else ""
                evidence = (f"{vcat} vocab ↔ {vattr} visual r={r:+.4f} "
                            f"p={p:.4f}{stars}")
                direction = "positive" if r > 0 else "negative"
                _add("text_visual", template, evidence,
                     round(r, 4), direction, "vocab_visual")

    # ── 3. Hypothesis-driven requirements ─────────────────────────────────
    if vocab_corr and vocab_corr.get("hypothesis_tests"):
        for test_key, test_data in vocab_corr["hypothesis_tests"].items():
            templates = HYPOTHESIS_REQ_TEMPLATES.get(test_key, {})
            supported = test_data.get("supports_hypothesis", False)
            template = templates.get(supported)
            if template:
                # Build evidence string from test data
                if "r" in test_data:
                    evidence = (f"r={test_data['r']:+.4f}, "
                                f"p={test_data.get('p', 'N/A')}")
                else:
                    evidence = (f"mean_diff={test_data.get('difference', 0):+.4f}, "
                                f"supports={supported}")
                _add("hypothesis", template, evidence,
                     test_data.get("r", test_data.get("difference", 0)),
                     "supports" if supported else "refutes",
                     "hypothesis_test")

    # ── 4. Always add a baseline requirement ──────────────────────────────
    if not requirements:
        _add(
            "baseline",
            "Insufficient correlation data for specific requirements — "
            "ensure equal numbers of images and text per country and re-run.",
            "n/a",
            0.0,
            "neutral",
            "fallback",
        )

    # ── 5. Build DataFrame and save ───────────────────────────────────────
    req_df = pd.DataFrame(requirements)
    csv_path = OUTPUT_DIR / "design_requirements.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    req_df.to_csv(csv_path, index=False)
    print(f"\n  Generated {len(req_df)} design requirements → {csv_path}")

    for _, row in req_df.iterrows():
        print(f"    {row['id']}  [{row['category']}]  {row['requirement'][:90]}...")

    return req_df


# ── Main Analysis Function ─────────────────────────────────────────────────────

def cross_analyze():
    """Combine and cross-analyze all pipeline results."""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CROSS_DIR.mkdir(parents=True, exist_ok=True)
    
    dfs = {}
    
    if TEXT_CSV.exists():
        dfs['text'] = pd.read_csv(TEXT_CSV)
        print(f"Loaded {len(dfs['text'])} text records with sentiment")
    else:
        print(f"WARNING: {TEXT_CSV} not found. Run 03_sentiment_analysis.py first.")
    
    if SENTIMENT_CSV.exists():
        dfs['sentiment_summary'] = pd.read_csv(SENTIMENT_CSV, index_col=0)
        print("Loaded sentiment summary")
    
    if IMAGE_CSV.exists():
        dfs['image'] = pd.read_csv(IMAGE_CSV)
        print(f"Loaded {len(dfs['image'])} image analyses")
    else:
        print(f"WARNING: {IMAGE_CSV} not found. Run 04_image_processing.py first.")
    
    if not dfs:
        print("ERROR: No data sources found. Run previous pipeline stages first.")
        return
    
    # ── Generate Summary Statistics ─────────────────────────────────────────────
    summary = {"analyzed_at": datetime.now().isoformat()}
    
    if 'text' in dfs:
        df = dfs['text']
        summary["total_text_records"] = len(df)
        summary["countries_with_data"] = df['country'].nunique() if 'country' in df.columns else 0
        
        if 'sentiment' in df.columns:
            summary["mean_sentiment"] = round(df['sentiment'].mean(), 3)
            summary["positive_ratio"] = round((df['sentiment'] == 1).mean() * 100, 1)
            summary["neutral_ratio"] = round((df['sentiment'] == 0).mean() * 100, 1)
            summary["negative_ratio"] = round((df['sentiment'] == -1).mean() * 100, 1)
    
    if 'image' in dfs:
        img_df = dfs['image']
        summary["total_images"] = len(img_df)
        summary["avg_image_width"] = round(img_df['width'].mean(), 0) if 'width' in img_df.columns else 0
        summary["avg_image_height"] = round(img_df['height'].mean(), 0) if 'height' in img_df.columns else 0
        summary["total_image_size_mb"] = round(img_df['file_size_kb'].sum() / 1024, 2) if 'file_size_kb' in img_df.columns else 0
        
        if 'country' in img_df.columns:
            summary["countries_with_images"] = img_df['country'].nunique()

    # ── Human grading metrics (optional) ────────────────────────────────────
    human_grades_path = DATA_ROOT / "human_grades.csv"
    if human_grades_path.exists():
        import pandas as pd
        hgrades = pd.read_csv(human_grades_path)
        summary["human_grades_total"] = len(hgrades)
        if len(hgrades) > 0:
            grade_sample_path = DATA_ROOT / "grade_sample.csv"
            if grade_sample_path.exists():
                gsample = pd.read_csv(grade_sample_path)
                gsample["snippet_id"] = gsample["snippet_id"].astype(str)
                hgrades["snippet_id"] = hgrades["snippet_id"].astype(str)
                hmerged = hgrades.merge(gsample[["snippet_id", "country", "label"]], on="snippet_id", how="left")
                hmerged["human_label"] = hmerged["human_score"].map({0: "NEG", 1: "NEU", 2: "POS"})
                hmerged["ai_short"] = hmerged["label"].apply(
                    lambda x: "POS" if "POS" in str(x).upper() else ("NEG" if "NEG" in str(x).upper() else "NEU"))
                hmerged["agree"] = hmerged["human_label"] == hmerged["ai_short"]
                summary["human_ai_agreement_rate"] = round(hmerged["agree"].mean() * 100, 1)
                if "country" in hmerged.columns:
                    for c in hmerged["country"].dropna().unique():
                        sub = hmerged[hmerged["country"] == c]
                        if len(sub) > 0:
                            summary[f"human_ai_agreement_{c}"] = round(sub["agree"].mean() * 100, 1)
    
    if 'text' in dfs and 'image' in dfs:
        text_countries = set(dfs['text'].get('country', pd.Series()).dropna().unique())
        image_countries = set(dfs['image'].get('country', pd.Series()).dropna().unique())
        summary["countries_with_both"] = len(text_countries & image_countries)
        summary["coverage_balance_score"] = round(
            len(text_countries & image_countries) / max(len(text_countries | image_countries), 1) * 100, 1
        )
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved analysis summary to {SUMMARY_CSV}")
    
    print("\n=== Cross Analysis Summary ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # ── Generate Cross Analysis by Country ─────────────────────────────────────
    if 'text' in dfs and 'image' in dfs:
        text_by_country = dfs['text'].groupby('country').agg({
            'sentiment': ['mean', 'count'],
            'confidence': 'mean' if 'confidence' in dfs['text'].columns else 'size'
        }).reset_index()
        text_by_country.columns = ['country', 'avg_sentiment', 'text_count', 'avg_confidence']
        
        img_by_country = dfs['image'].groupby('country').agg({
            'filename': 'count',
            'file_size_kb': 'sum'
        }).reset_index()
        img_by_country.columns = ['country', 'image_count', 'total_image_size_kb']
        
        cross = text_by_country.merge(img_by_country, on='country', how='outer')
        cross = cross.fillna(0)
        
        cross.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved cross-analysis by country to {OUTPUT_CSV}")
        print("\n=== Cross Analysis by Country ===")
        print(cross.to_string())
    
    # ── Compute Design–Sentiment Correlation Model ──────────────────────────────
    text_df = dfs.get('text', pd.DataFrame())
    img_df = dfs.get('image', pd.DataFrame())
    weights_data = None

    if len(text_df) > 0 and len(img_df) > 0:
        weights_data = compute_design_weights(text_df, img_df)
    else:
        print("\n  ⚠ Insufficient data for design-weight correlation model "
              "(need both text and image data)")

    # ── Generate All Visualizations ─────────────────────────────────────────────
    print("\n=== Generating Cross Analysis Visualizations ===")
    
    sentiment_summary = dfs.get('sentiment_summary')
    
    if len(text_df) > 0 and len(img_df) > 0:
        plot_text_vs_image_by_country(text_df, img_df, CROSS_DIR)
        plot_sentiment_vs_image_volume(text_df, img_df, CROSS_DIR)
        plot_combined_country_summary(text_df, img_df, sentiment_summary, CROSS_DIR)
        plot_balance_ratio_chart(text_df, img_df, CROSS_DIR)
        plot_coverage_summary(text_df, img_df, CROSS_DIR)
        plot_sentiment_heatmap(text_df, CROSS_DIR)
        plot_human_ai_agreement(text_df, CROSS_DIR)

        # Figure 8: correlation model heatmap (requires design weights)
        if weights_data is not None:
            plot_correlation_heatmap(weights_data, CROSS_DIR)

        # Vocabulary × Visual attribute correlation
        vocab_corr = None
        if len(text_df) > 0 and len(img_df) > 0:
            vocab_corr = compute_vocab_visual_correlation(text_df, img_df)
            plot_vocab_visual_heatmap(vocab_corr, CROSS_DIR)

        # Auto-generate design requirements from all correlation evidence
        generate_design_requirements(weights_data, vocab_corr)
    else:
        print("  ⚠ Insufficient data for visualizations")
    
    print("\n" + "=" * 50)
    print("✓ Cross Analysis Complete")
    print("=" * 50)
    print(f"\nGenerated up to 8 visualization figures in: {CROSS_DIR}")
    print("  • Figure 1: text_vs_image_by_country.png")
    print("  • Figure 2: sentiment_vs_image_volume.png")
    print("  • Figure 3: combined_country_summary.png")
    print("  • Figure 4: balance_ratio_chart.png")
    print("  • Figure 5: coverage_summary.png")
    print("  • Figure 6: sentiment_heatmap.png")
    print("  • Figure 7: human_ai_agreement.png (when human grades exist)")
    print("  • Figure 8: correlation_heatmap.png (design-weight model)")
    print("  • Figure 9: vocab_visual_heatmap.png (text × visual correlation)")
    if weights_data is not None:
        print(f"\n  Design weights model saved to: {WEIGHTS_JSON}")
        print(f"  Vocab-visual correlation saved to: {OUTPUT_DIR / 'vocab_visual_correlation.json'}")
        print(f"  Design requirements saved to: {OUTPUT_DIR / 'design_requirements.csv'}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    print("=== Starting Cross Analysis ===")
    cross_analyze()
    print("=== Cross Analysis Complete ===")