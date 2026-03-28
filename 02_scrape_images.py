"""
02_scrape_images.py — Scrape images of manhole covers
"""

import os
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
IMAGE_DIR = "/data/images/"
REDDIT_CSV = "/data/reddit_raw.csv"
OUTPUT_CSV = "/data/images_metadata.csv"

# ── Configuration ─────────────────────────────────────────────────────────────
USER_AGENT = "manhole-cover-scraper/1.0"
TIMEOUT = 10

def download_image(url, save_path):
    """Download an image from a URL."""
    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        
        with open(save_path, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        return False

def is_image_url(url):
    """Check if URL appears to be an image."""
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]
    url_lower = url.lower()
    return any(ext in url_lower for ext in image_extensions)

def scrape_images():
    """Download images from Reddit posts."""
    
    # Ensure image directory exists
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    # Load Reddit data
    if not os.path.exists(REDDIT_CSV):
        print(f"ERROR: {REDDIT_CSV} not found. Run 01_scrape_reddit.py first.")
        return
    
    df = pd.read_csv(REDDIT_CSV)
    print(f"Loaded {len(df)} posts from {REDDIT_CSV}")
    
    # Filter posts with image URLs
    image_posts = df[df["url"].apply(is_image_url)]
    print(f"Found {len(image_posts)} posts with image URLs")
    
    metadata = []
    downloaded = 0
    
    for idx, row in image_posts.iterrows():
        post_id = row["id"]
        url = row["url"]
        
        # Determine file extension
        ext = ".jpg"
        for e in [".png", ".gif", ".webp", ".jpeg", ".jpg"]:
            if e in url.lower():
                ext = e
                break
        
        filename = f"{post_id}{ext}"
        save_path = os.path.join(IMAGE_DIR, filename)
        
        print(f"Downloading {filename}...")
        if download_image(url, save_path):
            downloaded += 1
            metadata.append({
                "post_id": post_id,
                "filename": filename,
                "source_url": url,
                "title": row["title"],
                "subreddit": row["subreddit"],
                "downloaded_at": datetime.now().isoformat()
            })
    
    # Save metadata
    if metadata:
        meta_df = pd.DataFrame(metadata)
        meta_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved {len(metadata)} images to {IMAGE_DIR}")
        print(f"Metadata saved to {OUTPUT_CSV}")
    else:
        print("No images downloaded")

if __name__ == "__main__":
    print("=== Starting Image Scraper ===")
    scrape_images()
    print("=== Image Scraper Complete ===")