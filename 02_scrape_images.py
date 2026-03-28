"""
02_scrape_images.py — Scrape images from Reddit, Flickr, and Wikimedia Commons
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

# ── Flickr API Configuration (Optional) ───────────────────────────────────────
FLICKR_API_KEY = os.environ.get("FLICKR_API_KEY", "")

# ── Configuration ─────────────────────────────────────────────────────────────
USER_AGENT = "manhole-cover-scraper/1.0"
TIMEOUT = 30


def download_image(url, save_path):
    """Download an image from a URL."""
    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=TIMEOUT, stream=True)
        response.raise_for_status()
        
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    Failed to download {url}: {e}")
        return False


def is_image_url(url):
    """Check if URL appears to be an image."""
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]
    url_lower = url.lower()
    return any(ext in url_lower for ext in image_extensions)


def scrape_from_reddit_csv():
    """Download images from URLs in the Reddit CSV (includes Wikimedia entries)."""
    images = []
    
    if not os.path.exists(REDDIT_CSV):
        print(f"  {REDDIT_CSV} not found - skipping CSV images")
        return images
    
    df = pd.read_csv(REDDIT_CSV)
    print(f"  Loaded {len(df)} entries from {REDDIT_CSV}")
    
    # Filter entries with image URLs
    image_posts = df[df["url"].apply(lambda x: is_image_url(str(x)) if pd.notna(x) else False)]
    print(f"  Found {len(image_posts)} entries with image URLs")
    
    for idx, row in image_posts.iterrows():
        post_id = row["id"]
        url = str(row["url"])
        
        # Determine file extension
        ext = ".jpg"
        for e in [".png", ".gif", ".webp", ".jpeg", ".jpg"]:
            if e in url.lower():
                ext = e
                break
        
        # Create safe filename
        safe_id = post_id.replace("/", "_").replace(":", "_")
        filename = f"{safe_id}{ext}"
        
        images.append({
            "post_id": post_id,
            "filename": filename,
            "source_url": url,
            "title": row.get("title", "")[:100] if pd.notna(row.get("title")) else "",
            "source": row.get("source", "csv"),
            "from_csv": True
        })
    
    return images


def scrape_flickr():
    """Scrape images from Flickr. Requires API key."""
    images = []
    
    if not FLICKR_API_KEY:
        print("  Flickr API key not configured - skipping Flickr")
        return images
    
    print("  Searching Flickr...")
    
    base_url = "https://www.flickr.com/services/rest/"
    
    search_terms = ["manhole cover", "manhole art", "japanese manhole"]
    
    for term in search_terms:
        try:
            params = {
                "method": "flickr.photos.search",
                "api_key": FLICKR_API_KEY,
                "text": term,
                "license": "1,2,3,4,5,6,7,8,9,10",  # Various CC licenses
                "sort": "relevance",
                "per_page": 50,
                "format": "json",
                "nojsoncallback": 1,
                "extras": "url_m,owner_name,title"
            }
            
            response = requests.get(base_url, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            if data.get("stat") != "ok":
                print(f"    Flickr API error: {data.get('message', 'Unknown error')}")
                continue
            
            photos = data.get("photos", {}).get("photo", [])
            
            for photo in photos:
                photo_id = photo.get("id")
                url = photo.get("url_m")
                
                if not url:
                    continue
                
                images.append({
                    "post_id": f"flickr_{photo_id}",
                    "filename": f"flickr_{photo_id}.jpg",
                    "source_url": url,
                    "title": photo.get("title", "")[:100],
                    "source": "flickr",
                    "from_csv": False
                })
                
        except Exception as e:
            print(f"    Error searching Flickr for '{term}': {e}")
    
    print(f"    Found {len(images)} Flickr images")
    return images


def scrape_wikimedia_direct():
    """Scrape images directly from Wikimedia Commons API. No API key required."""
    images = []
    
    print("  Searching Wikimedia Commons directly...")
    
    base_url = "https://commons.wikimedia.org/w/api.php"
    
    search_terms = [
        "manhole cover",
        "manhole",
        "drain cover",
        "sewer cover",
        "Japanese manhole",
    ]
    
    for term in search_terms:
        try:
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
            
            response = requests.get(base_url, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            results = data.get("query", {}).get("search", [])
            
            if not results:
                continue
            
            # Get image URLs for all results
            pageids = [str(r["pageid"]) for r in results]
            
            info_params = {
                "action": "query",
                "pageids": "|".join(pageids),
                "prop": "imageinfo",
                "iiprop": "url|extmetadata",
                "iiurlwidth": 800,  # Get medium-sized images
                "format": "json",
                "formatversion": 2
            }
            
            info_response = requests.get(base_url, params=info_params, timeout=TIMEOUT)
            info_response.raise_for_status()
            info_data = info_response.json()
            
            pages = info_data.get("query", {}).get("pages", [])
            
            for page in pages:
                pageid = page.get("pageid")
                title = page.get("title", "").replace("File:", "")
                image_info = page.get("imageinfo", [{}])[0] if page.get("imageinfo") else {}
                
                # Get the medium-sized URL if available, otherwise full
                url = image_info.get("thumburl") or image_info.get("url", "")
                
                if not url:
                    continue
                
                # Determine extension
                ext = ".jpg"
                for e in [".png", ".gif", ".webp", ".jpeg", ".jpg"]:
                    if e in url.lower():
                        ext = e
                        break
                
                images.append({
                    "post_id": f"wikimedia_{pageid}",
                    "filename": f"wikimedia_{pageid}{ext}",
                    "source_url": url,
                    "title": title[:100],
                    "source": "wikimedia_commons",
                    "from_csv": False
                })
                
        except Exception as e:
            print(f"    Error searching Wikimedia Commons for '{term}': {e}")
    
    print(f"    Found {len(images)} Wikimedia Commons images")
    return images


def main():
    """Run all image scrapers and download images."""
    
    # Ensure image directory exists
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    all_images = []
    
    # Get images from Reddit CSV (includes any Wikimedia entries already scraped)
    print("Checking CSV for image URLs...")
    csv_images = scrape_from_reddit_csv()
    all_images.extend(csv_images)
    
    # Try Flickr (optional)
    print("Checking Flickr...")
    try:
        flickr_images = scrape_flickr()
        all_images.extend(flickr_images)
    except Exception as e:
        print(f"  Flickr scraping failed: {e}")
    
    # Wikimedia Commons direct (no API key required) - additional source
    print("Checking Wikimedia Commons directly...")
    try:
        wikimedia_images = scrape_wikimedia_direct()
        all_images.extend(wikimedia_images)
    except Exception as e:
        print(f"  Wikimedia Commons direct scraping failed: {e}")
    
    # Remove duplicates by filename
    seen_filenames = set()
    unique_images = []
    for img in all_images:
        if img["filename"] not in seen_filenames:
            seen_filenames.add(img["filename"])
            unique_images.append(img)
    
    print(f"\nTotal unique images to download: {len(unique_images)}")
    
    if not unique_images:
        print("No images to download")
        return
    
    # Download images
    metadata = []
    downloaded = 0
    
    for idx, img in enumerate(unique_images):
        save_path = os.path.join(IMAGE_DIR, img["filename"])
        
        # Skip if already exists
        if os.path.exists(save_path):
            print(f"  [{idx + 1}/{len(unique_images)}] Already exists: {img['filename']}")
            metadata.append({
                "post_id": img["post_id"],
                "filename": img["filename"],
                "source_url": img["source_url"],
                "title": img.get("title", ""),
                "source": img.get("source", "unknown"),
                "downloaded_at": datetime.now().isoformat()
            })
            downloaded += 1
            continue
        
        print(f"  [{idx + 1}/{len(unique_images)}] Downloading: {img['filename']}...")
        if download_image(img["source_url"], save_path):
            downloaded += 1
            metadata.append({
                "post_id": img["post_id"],
                "filename": img["filename"],
                "source_url": img["source_url"],
                "title": img.get("title", ""),
                "source": img.get("source", "unknown"),
                "downloaded_at": datetime.now().isoformat()
            })
    
    # Save metadata
    if metadata:
        meta_df = pd.DataFrame(metadata)
        meta_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nDownloaded {downloaded} images to {IMAGE_DIR}")
        print(f"Metadata saved to {OUTPUT_CSV}")
    else:
        print("\nNo images downloaded")


if __name__ == "__main__":
    print("=== Starting Image Scraper ===")
    main()
    print("=== Image Scraper Complete ===")