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
    
    # ── Generate All Visualizations ─────────────────────────────────────────────
    print("\n=== Generating Cross Analysis Visualizations ===")
    
    text_df = dfs.get('text', pd.DataFrame())
    img_df = dfs.get('image', pd.DataFrame())
    sentiment_summary = dfs.get('sentiment_summary')
    
    if len(text_df) > 0 and len(img_df) > 0:
        plot_text_vs_image_by_country(text_df, img_df, CROSS_DIR)
        plot_sentiment_vs_image_volume(text_df, img_df, CROSS_DIR)
        plot_combined_country_summary(text_df, img_df, sentiment_summary, CROSS_DIR)
        plot_balance_ratio_chart(text_df, img_df, CROSS_DIR)
        plot_coverage_summary(text_df, img_df, CROSS_DIR)
        plot_sentiment_heatmap(text_df, CROSS_DIR)
    else:
        print("  ⚠ Insufficient data for visualizations")
    
    print("\n" + "=" * 50)
    print("✓ Cross Analysis Complete")
    print("=" * 50)
    print(f"\nGenerated 6 visualization figures in: {CROSS_DIR}")
    print("  • Figure 1: text_vs_image_by_country.png")
    print("  • Figure 2: sentiment_vs_image_volume.png")
    print("  • Figure 3: combined_country_summary.png")
    print("  • Figure 4: balance_ratio_chart.png")
    print("  • Figure 5: coverage_summary.png")
    print("  • Figure 6: sentiment_heatmap.png")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    print("=== Starting Cross Analysis ===")
    cross_analyze()
    print("=== Cross Analysis Complete ===")