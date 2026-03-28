"""
01_scrape_reddit.py — Scrape Reddit for manhole cover discussions
"""

import os
import praw
import pandas as pd
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_PATH = "/data/reddit_raw.csv"

# ── Reddit API Configuration ─────────────────────────────────────────────────
# Set these environment variables in Zeabur
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "manhole-cover-scraper/1.0")

def scrape_reddit():
    """Scrape Reddit for posts about manhole covers."""
    
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        print("ERROR: Reddit API credentials not configured")
        print("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables")
        return
    
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    
    # Subreddits to search
    subreddits = ["japan", "tokyo", "urbanexploration", "infrastructure"]
    
    # Search terms
    search_terms = ["manhole cover", "manhole", "drain cover", "道路蓋"]
    
    posts = []
    
    for sub_name in subreddits:
        try:
            subreddit = reddit.subreddit(sub_name)
            for term in search_terms:
                print(f"Searching r/{sub_name} for '{term}'...")
                for post in subreddit.search(term, limit=50, sort="relevance"):
                    posts.append({
                        "id": post.id,
                        "title": post.title,
                        "selftext": post.selftext,
                        "author": str(post.author) if post.author else "[deleted]",
                        "subreddit": str(post.subreddit),
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "created_utc": datetime.fromtimestamp(post.created_utc).isoformat(),
                        "url": post.url,
                        "permalink": f"https://reddit.com{post.permalink}",
                        "search_term": term
                    })
        except Exception as e:
            print(f"Error searching r/{sub_name}: {e}")
    
    # Remove duplicates
    df = pd.DataFrame(posts)
    if not df.empty:
        df = df.drop_duplicates(subset=["id"])
        print(f"Found {len(df)} unique posts")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        
        # Save to CSV
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved to {OUTPUT_PATH}")
    else:
        print("No posts found")

if __name__ == "__main__":
    print("=== Starting Reddit Scraper ===")
    scrape_reddit()
    print("=== Reddit Scraper Complete ===")