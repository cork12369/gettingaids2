"""
STAGE 4: Cross Analysis
Cross-analyze sentiment and image data from all pipeline stages.

Reads from:
  /data/text_with_sentiment.csv  (from Stage 2)
  /data/output/image_analysis.csv (from Stage 3)

Outputs:
  /data/output/cross_analysis.csv
  /data/output/analysis_summary.csv
"""

import os
import pandas as pd
from datetime import datetime
from pathlib import Path

# Paths
TEXT_CSV        = Path("/data/text_with_sentiment.csv")
SENTIMENT_CSV   = Path("/data/output/sentiment_summary.csv")
IMAGE_CSV       = Path("/data/output/image_analysis.csv")
OUTPUT_DIR      = Path("/data/output")
OUTPUT_CSV      = OUTPUT_DIR / "cross_analysis.csv"
SUMMARY_CSV     = OUTPUT_DIR / "analysis_summary.csv"


def cross_analyze():
    """Combine and cross-analyze all pipeline results."""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    dfs = {}
    
    if TEXT_CSV.exists():
        dfs['text'] = pd.read_csv(TEXT_CSV)
        print(f"Loaded {len(dfs['text'])} text records with sentiment")
    else:
        print(f"WARNING: {TEXT_CSV} not found. Run 03_sentiment_analysis.py first.")
    
    if SENTIMENT_CSV.exists():
        dfs['sentiment_summary'] = pd.read_csv(SENTIMENT_CSV)
        print("Loaded sentiment summary")
    
    if IMAGE_CSV.exists():
        dfs['image'] = pd.read_csv(IMAGE_CSV)
        print(f"Loaded {len(dfs['image'])} image analyses")
    else:
        print(f"WARNING: {IMAGE_CSV} not found. Run 04_image_processing.py first.")
    
    if not dfs:
        print("ERROR: No data sources found. Run previous pipeline stages first.")
        return
    
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
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved analysis summary to {SUMMARY_CSV}")
    
    print("\n=== Cross Analysis Summary ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
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
    
    print("\nCross analysis complete")


if __name__ == "__main__":
    print("=== Starting Cross Analysis ===")
    cross_analyze()
    print("=== Cross Analysis Complete ===")