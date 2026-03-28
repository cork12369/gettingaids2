"""
05_cross_analysis.py — Cross-analyze sentiment and image data
"""

import os
import pandas as pd
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
REDDIT_CSV = "/data/reddit_raw.csv"
SENTIMENT_CSV = "/data/output/sentiment_results.csv"
IMAGE_CSV = "/data/output/image_analysis.csv"
OUTPUT_DIR = "/data/output/"
OUTPUT_CSV = "/data/output/cross_analysis.csv"
SUMMARY_CSV = "/data/output/analysis_summary.csv"

def cross_analyze():
    """Combine and cross-analyze all pipeline results."""
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load all data sources
    dfs = {}
    
    if os.path.exists(REDDIT_CSV):
        dfs['reddit'] = pd.read_csv(REDDIT_CSV)
        print(f"Loaded {len(dfs['reddit'])} Reddit posts")
    else:
        print(f"WARNING: {REDDIT_CSV} not found")
    
    if os.path.exists(SENTIMENT_CSV):
        dfs['sentiment'] = pd.read_csv(SENTIMENT_CSV)
        print(f"Loaded {len(dfs['sentiment'])} sentiment results")
    else:
        print(f"WARNING: {SENTIMENT_CSV} not found")
    
    if os.path.exists(IMAGE_CSV):
        dfs['image'] = pd.read_csv(IMAGE_CSV)
        print(f"Loaded {len(dfs['image'])} image analyses")
    else:
        print(f"WARNING: {IMAGE_CSV} not found")
    
    if not dfs:
        print("ERROR: No data sources found. Run previous pipeline stages first.")
        return
    
    # Cross-analyze: merge Reddit + Sentiment + Image data
    merged = None
    
    if 'reddit' in dfs and 'sentiment' in dfs:
        # Merge Reddit posts with sentiment
        merged = dfs['reddit'].merge(
            dfs['sentiment'],
            left_on='id',
            right_on='post_id',
            how='left',
            suffixes=('', '_sent')
        )
        print(f"Merged Reddit + Sentiment: {len(merged)} rows")
    
    if merged is not None and 'image' in dfs:
        # Add image data
        merged = merged.merge(
            dfs['image'][['post_id', 'filename', 'width', 'height', 'file_size_kb']],
            left_on='id',
            right_on='post_id',
            how='left',
            suffixes=('', '_img')
        )
        print(f"Merged with Image data: {len(merged)} rows")
    
    if merged is not None:
        # Save merged results
        merged.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved cross-analysis to {OUTPUT_CSV}")
        
        # Create summary
        summary = {
            "total_posts": len(merged),
            "posts_with_sentiment": merged['dominant_sentiment'].notna().sum() if 'dominant_sentiment' in merged else 0,
            "posts_with_images": merged['filename'].notna().sum() if 'filename' in merged else 0,
            "positive_posts": (merged['dominant_sentiment'] == 'positive').sum() if 'dominant_sentiment' in merged else 0,
            "neutral_posts": (merged['dominant_sentiment'] == 'neutral').sum() if 'dominant_sentiment' in merged else 0,
            "negative_posts": (merged['dominant_sentiment'] == 'negative').sum() if 'dominant_sentiment' in merged else 0,
            "analyzed_at": datetime.now().isoformat()
        }
        
        # Add subreddit breakdown if available
        if 'subreddit' in merged:
            summary["subreddits"] = merged['subreddit'].nunique()
        
        # Add image stats if available
        if 'file_size_kb' in merged:
            summary["total_image_size_mb"] = round(merged['file_size_kb'].sum() / 1024, 2)
        
        # Save summary
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(SUMMARY_CSV, index=False)
        print(f"Saved analysis summary to {SUMMARY_CSV}")
        
        # Print summary
        print("\n=== Cross Analysis Summary ===")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    else:
        print("No data to cross-analyze")

if __name__ == "__main__":
    print("=== Starting Cross Analysis ===")
    cross_analyze()
    print("=== Cross Analysis Complete ===")