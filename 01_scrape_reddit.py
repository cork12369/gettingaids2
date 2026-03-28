"""
01_scrape_reddit.py — Scrape Reddit and/or Wikimedia Commons for manhole cover discussions
"""

import os
import requests
import pandas as pd
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_PATH = "/data/reddit_raw.csv"

# ── Reddit API Configuration (Optional) ───────────────────────────────────────
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "manhole-cover-scraper/1.0")

def scrape_reddit():
    """Scrape Reddit for posts about manhole covers. Requires API credentials."""
    import praw
    
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        print("Reddit API credentials not configured - skipping Reddit")
        return []
    
    print("Scraping Reddit...")
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    
    subreddits = ["japan", "tokyo", "urbanexploration", "infrastructure"]
    search_terms = ["manhole cover", "manhole", "drain cover"]
    
    posts = []
    
    for sub_name in subreddits:
        try:
            subreddit = reddit.subreddit(sub_name)
            for term in search_terms:
                print(f"  Searching r/{sub_name} for '{term}'...")
                for post in subreddit.search(term, limit=50, sort="relevance"):
                    posts.append({
                        "id": f"reddit_{post.id}",
                        "title": post.title,
                        "selftext": post.selftext,
                        "author": str(post.author) if post.author else "[deleted]",
                        "source": f"reddit/r/{sub_name}",
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "created_utc": datetime.fromtimestamp(post.created_utc).isoformat(),
                        "url": post.url,
                        "permalink": f"https://reddit.com{post.permalink}",
                        "search_term": term
                    })
        except Exception as e:
            print(f"  Error searching r/{sub_name}: {e}")
    
    print(f"  Found {len(posts)} Reddit posts")
    return posts


def scrape_wikimedia_commons():
    """Scrape Wikimedia Commons for manhole cover content. No API key required."""
    print("Scraping Wikimedia Commons...")
    
    # Wikimedia Commons API endpoint
    base_url = "https://commons.wikimedia.org/w/api.php"
    
    # Search terms for manhole covers
    search_terms = [
        "manhole cover",
        "manhole",
        "drain cover",
        "sewer cover",
        "日本 マンホール",  # Japanese manhole
    ]
    
    posts = []
    
    for term in search_terms:
        try:
            print(f"  Searching Wikimedia Commons for '{term}'...")
            
            # Search for images
            params = {
                "action": "query",
                "list": "search",
                "srsearch": term,
                "srnamespace": "6",  # File namespace
                "srlimit": 50,
                "format": "json",
                "formatversion": 2
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            results = data.get("query", {}).get("search", [])
            
            for item in results:
                title = item.get("title", "")
                pageid = item.get("pageid", 0)
                
                # Get image info
                info_params = {
                    "action": "query",
                    "pageids": pageid,
                    "prop": "imageinfo",
                    "iiprop": "url|extmetadata",
                    "format": "json",
                    "formatversion": 2
                }
                
                try:
                    info_response = requests.get(base_url, params=info_params, timeout=10)
                    info_response.raise_for_status()
                    info_data = info_response.json()
                    
                    pages = info_data.get("query", {}).get("pages", [])
                    if pages:
                        page = pages[0]
                        image_info = page.get("imageinfo", [{}])[0] if page.get("imageinfo") else {}
                        metadata = image_info.get("extmetadata", {})
                        
                        posts.append({
                            "id": f"wikimedia_{pageid}",
                            "title": title.replace("File:", ""),
                            "selftext": metadata.get("ImageDescription", {}).get("value", ""),
                            "author": metadata.get("Artist", {}).get("value", "Unknown"),
                            "source": "wikimedia_commons",
                            "score": 0,
                            "num_comments": 0,
                            "created_utc": metadata.get("DateTime", {}).get("value", datetime.now().isoformat()),
                            "url": image_info.get("url", ""),
                            "permalink": f"https://commons.wikimedia.org/?curid={pageid}",
                            "search_term": term
                        })
                except Exception as e:
                    print(f"    Error getting info for {title}: {e}")
                    continue
                    
        except Exception as e:
            print(f"  Error searching Wikimedia Commons for '{term}': {e}")
    
    print(f"  Found {len(posts)} Wikimedia Commons entries")
    return posts


def main():
    """Run all scrapers and combine results."""
    all_posts = []
    
    # Try Reddit (optional - requires API keys)
    try:
        reddit_posts = scrape_reddit()
        all_posts.extend(reddit_posts)
    except Exception as e:
        print(f"Reddit scraping failed: {e}")
    
    # Wikimedia Commons (no API key required)
    try:
        wikimedia_posts = scrape_wikimedia_commons()
        all_posts.extend(wikimedia_posts)
    except Exception as e:
        print(f"Wikimedia Commons scraping failed: {e}")
    
    # Save results
    if all_posts:
        df = pd.DataFrame(all_posts)
        df = df.drop_duplicates(subset=["id"])
        print(f"\nTotal unique entries: {len(df)}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved to {OUTPUT_PATH}")
    else:
        print("\nNo posts found from any source")
        # Create empty CSV with headers
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        pd.DataFrame(columns=[
            "id", "title", "selftext", "author", "source", "score",
            "num_comments", "created_utc", "url", "permalink", "search_term"
        ]).to_csv(OUTPUT_PATH, index=False)
        print(f"Created empty {OUTPUT_PATH}")


if __name__ == "__main__":
    print("=== Starting Data Scraper ===")
    main()
    print("=== Data Scraper Complete ===")