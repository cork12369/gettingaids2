"""
STAGE 5: Image Visualization Generator
Generates PNG charts from image_analysis.csv (Stage 3 output).

This script is called after Stage 3 to generate the 4 PNG charts expected
by the dashboard in the Image Analysis section.

Reads from:
  /data/output/image_analysis.csv  (from Stage 3)

Outputs to:
  /data/output/image_volume_by_country.png
  /data/output/country_image_distribution.png
  /data/output/source_distribution.png
  /data/output/image_resolution_distribution.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
OUTPUT_DIR = DATA_DIR / "output"
IMAGE_ANALYSIS_CSV = OUTPUT_DIR / "image_analysis.csv"

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_image_volume_by_country(df, output_dir):
    """Figure 1: Image count by country (horizontal bar chart)"""
    if 'country' not in df.columns:
        print("  ⚠ 'country' column not found")
        return
    
    country_counts = df['country'].value_counts().sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(country_counts) * 0.5)))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(country_counts)))
    bars = ax.barh(country_counts.index, country_counts.values, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Number of Images', fontsize=11)
    ax.set_ylabel('Country', fontsize=11)
    ax.set_title('Image Volume by Country\n(From Scraped Data)', fontsize=13, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, country_counts.values):
        ax.text(val + max(country_counts.values) * 0.02, bar.get_y() + bar.get_height() / 2,
               str(val), va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'image_volume_by_country.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: image_volume_by_country.png")


def plot_country_image_distribution(df, output_dir):
    """Figure 2: Country distribution with percentage (pie chart)"""
    if 'country' not in df.columns:
        print("  ⚠ 'country' column not found")
        return
    
    country_counts = df['country'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = plt.cm.Set3(range(len(country_counts)))
    explode = [0.02] * len(country_counts)
    
    wedges, texts, autotexts = ax.pie(
        country_counts.values,
        labels=country_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        explode=explode,
        startangle=90,
        pctdistance=0.75
    )
    
    # Style percentage text
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    ax.set_title('Country Image Distribution\n(Percentage Share)', fontsize=13, fontweight='bold')
    
    # Add legend
    ax.legend(country_counts.index, country_counts.values, 
              title="Images", loc="center left", 
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'country_image_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: country_image_distribution.png")


def plot_source_distribution(df, output_dir):
    """Figure 3: Image source distribution (bar + pie combo)"""
    if 'source' not in df.columns:
        # Fallback: count images by filename pattern or parent folder
        if 'filename' in df.columns:
            sources = df['filename'].apply(lambda x: 'scraped' if 'ddg' in str(x).lower() else 'other')
        else:
            print("  ⚠ 'source' column not found, skipping source distribution")
            return
    
    source_counts = df['source'].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = axes[0].bar(source_counts.index, source_counts.values, color=colors[:len(source_counts)], 
                       edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Source', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Image Sources (Bar)', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, source_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Pie chart
    colors_pie = colors[:len(source_counts)]
    wedges, texts, autotexts = axes[1].pie(
        source_counts.values,
        labels=source_counts.index,
        autopct='%1.1f%%',
        colors=colors_pie,
        startangle=90,
        explode=[0.02] * len(source_counts)
    )
    axes[1].set_title('Image Sources (Pie)', fontsize=12, fontweight='bold')
    
    plt.suptitle('Source Distribution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'source_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: source_distribution.png")


def plot_resolution_distribution(df, output_dir):
    """Figure 4: Image resolution distribution (multi-panel)"""
    if 'width' not in df.columns or 'height' not in df.columns:
        print("  ⚠ 'width'/'height' columns not found")
        return
    
    df_clean = df.dropna(subset=['width', 'height'])
    if len(df_clean) == 0:
        print("  ⚠ No valid width/height data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Width distribution
    axes[0, 0].hist(df_clean['width'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Width (px)', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].set_title('Width Distribution', fontsize=11, fontweight='bold')
    axes[0, 0].axvline(df_clean['width'].median(), color='red', linestyle='--', label=f'Median: {df_clean["width"].median():.0f}')
    axes[0, 0].legend(fontsize=9)
    
    # 2. Height distribution
    axes[0, 1].hist(df_clean['height'], bins=30, color='#2ecc71', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Height (px)', fontsize=10)
    axes[0, 1].set_ylabel('Frequency', fontsize=10)
    axes[0, 1].set_title('Height Distribution', fontsize=11, fontweight='bold')
    axes[0, 1].axvline(df_clean['height'].median(), color='red', linestyle='--', label=f'Median: {df_clean["height"].median():.0f}')
    axes[0, 1].legend(fontsize=9)
    
    # 3. Scatter plot width vs height
    axes[1, 0].scatter(df_clean['width'], df_clean['height'], alpha=0.5, c='#9b59b6', edgecolor='black', linewidth=0.3)
    axes[1, 0].set_xlabel('Width (px)', fontsize=10)
    axes[1, 0].set_ylabel('Height (px)', fontsize=10)
    axes[1, 0].set_title('Width vs Height', fontsize=11, fontweight='bold')
    axes[1, 0].plot([0, df_clean['width'].max()], [0, df_clean['height'].max()], 'r--', alpha=0.5, label='1:1 line')
    axes[1, 0].legend(fontsize=9)
    
    # 4. Resolution category pie
    df_clean = df_clean.copy()
    df_clean['area'] = df_clean['width'] * df_clean['height']
    
    def categorize_res(area):
        area_m = area / 1_000_000  # megapixels
        if area_m < 0.5:
            return 'Low (<0.5 MP)'
        elif area_m < 2:
            return 'Medium (0.5-2 MP)'
        elif area_m < 5:
            return 'High (2-5 MP)'
        else:
            return 'Very High (>5 MP)'
    
    df_clean['resolution_category'] = df_clean['area'].apply(categorize_res)
    res_counts = df_clean['resolution_category'].value_counts()
    
    colors_res = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    axes[1, 1].pie(res_counts.values, labels=res_counts.index, autopct='%1.1f%%',
                   colors=colors_res[:len(res_counts)], startangle=90)
    axes[1, 1].set_title('Resolution Category', fontsize=11, fontweight='bold')
    
    plt.suptitle('Image Resolution Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'image_resolution_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: image_resolution_distribution.png")


def main():
    """Main function to generate all image visualization charts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not IMAGE_ANALYSIS_CSV.exists():
        print(f"ERROR: {IMAGE_ANALYSIS_CSV} not found.")
        print("Run Stage 3 (04_image_processing.py) first to generate image analysis data.")
        return
    
    print(f"Loading image analysis data from {IMAGE_ANALYSIS_CSV}...")
    df = pd.read_csv(IMAGE_ANALYSIS_CSV)
    print(f"  Loaded {len(df)} image records\n")
    
    print("Generating visualization charts...")
    
    plot_image_volume_by_country(df, OUTPUT_DIR)
    plot_country_image_distribution(df, OUTPUT_DIR)
    plot_source_distribution(df, OUTPUT_DIR)
    plot_resolution_distribution(df, OUTPUT_DIR)
    
    print(f"\n{'='*55}")
    print("  Stage 5 Complete (Image Visualizations)")
    print(f"{'='*55}")
    print(f"  Images processed : {len(df)}")
    print(f"  Countries       : {df['country'].nunique() if 'country' in df.columns else 'N/A'}")
    print(f"  Output dir      : {OUTPUT_DIR}")
    print(f"\n  Charts saved:")
    print(f"    • image_volume_by_country.png")
    print(f"    • country_image_distribution.png")
    print(f"    • source_distribution.png")
    print(f"    • image_resolution_distribution.png")


if __name__ == "__main__":
    main()
