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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from pathlib import Path

# Paths
TEXT_CSV        = Path("/data/text_with_sentiment.csv")
SENTIMENT_CSV   = Path("/data/output/sentiment_summary.csv")
IMAGE_CSV       = Path("/data/output/image_analysis.csv")
OUTPUT_DIR      = Path("/data/output")
CROSS_DIR       = Path("/data/output/cross_analysis_visualizations")
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
    human_grades_path = Path("/data/human_grades.csv")
    grade_sample_path = Path("/data/grade_sample.csv")
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
    human_grades_path = Path("/data/human_grades.csv")
    if human_grades_path.exists():
        import pandas as pd
        hgrades = pd.read_csv(human_grades_path)
        summary["human_grades_total"] = len(hgrades)
        if len(hgrades) > 0:
            grade_sample_path = Path("/data/grade_sample.csv")
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
    if weights_data is not None:
        print(f"\n  Design weights model saved to: {WEIGHTS_JSON}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    print("=== Starting Cross Analysis ===")
    cross_analyze()
    print("=== Cross Analysis Complete ===")