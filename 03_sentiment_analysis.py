"""
03_sentiment_analysis.py — Analyze sentiment of Reddit posts
"""

import os
import pandas as pd
from datetime import datetime

# ── Model Caching (for Hugging Face) ──────────────────────────────────────────
os.environ["TRANSFORMERS_CACHE"] = "/data/hf_cache"

from transformers import pipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
REDDIT_CSV = "/data/reddit_raw.csv"
OUTPUT_CSV = "/data/output/sentiment_results.csv"

# ── Configuration ─────────────────────────────────────────────────────────────
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def analyze_sentiment():
    """Analyze sentiment of Reddit posts."""
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # Ensure HF cache directory exists
    os.makedirs("/data/hf_cache", exist_ok=True)
    
    # Load Reddit data
    if not os.path.exists(REDDIT_CSV):
        print(f"ERROR: {REDDIT_CSV} not found. Run 01_scrape_reddit.py first.")
        return
    
    df = pd.read_csv(REDDIT_CSV)
    print(f"Loaded {len(df)} posts from {REDDIT_CSV}")
    
    # Initialize sentiment pipeline
    print(f"Loading sentiment model: {SENTIMENT_MODEL}")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        top_k=None  # Return all sentiment scores
    )
    
    results = []
    
    for idx, row in df.iterrows():
        # Combine title and selftext for analysis
        text = f"{row['title']} {row['selftext']}"
        
        # Truncate if too long (model limit is typically 512 tokens)
        if len(text) > 1000:
            text = text[:1000]
        
        print(f"Analyzing post {idx + 1}/{len(df)}: {row['title'][:50]}...")
        
        try:
            predictions = sentiment_pipeline(text)[0]
            
            # Extract scores
            scores = {p['label']: p['score'] for p in predictions}
            
            # Determine dominant sentiment
            dominant = max(predictions, key=lambda x: x['score'])
            
            results.append({
                "post_id": row['id'],
                "title": row['title'][:100],
                "positive_score": scores.get('positive', 0),
                "neutral_score": scores.get('neutral', 0),
                "negative_score": scores.get('negative', 0),
                "dominant_sentiment": dominant['label'],
                "confidence": dominant['score'],
                "analyzed_at": datetime.now().isoformat()
            })
        except Exception as e:
            print(f"  Error analyzing post {row['id']}: {e}")
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved {len(results)} sentiment results to {OUTPUT_CSV}")
        
        # Print summary
        print("\n=== Sentiment Summary ===")
        print(results_df['dominant_sentiment'].value_counts())
    else:
        print("No results to save")

if __name__ == "__main__":
    print("=== Starting Sentiment Analysis ===")
    analyze_sentiment()
    print("=== Sentiment Analysis Complete ===")