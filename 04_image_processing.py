"""
04_image_processing.py — Process and analyze manhole cover images
"""

import os
import pandas as pd
from datetime import datetime
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
IMAGE_DIR = "/data/images/"
IMAGE_METADATA = "/data/images_metadata.csv"
OUTPUT_DIR = "/data/output/"
OUTPUT_CSV = "/data/output/image_analysis.csv"

# ── Configuration ─────────────────────────────────────────────────────────────
SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]

def get_image_info(image_path):
    """Extract basic information from an image."""
    try:
        with Image.open(image_path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "aspect_ratio": round(img.width / img.height, 2) if img.height > 0 else 0
            }
    except Exception as e:
        print(f"  Error reading {image_path}: {e}")
        return None

def process_images():
    """Process and analyze downloaded images."""
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if image directory exists
    if not os.path.exists(IMAGE_DIR):
        print(f"ERROR: {IMAGE_DIR} not found. Run 02_scrape_images.py first.")
        return
    
    # Get all image files
    image_files = []
    for f in os.listdir(IMAGE_DIR):
        if any(f.lower().endswith(ext) for ext in SUPPORTED_FORMATS):
            image_files.append(f)
    
    print(f"Found {len(image_files)} images in {IMAGE_DIR}")
    
    # Load metadata if available
    metadata_df = None
    if os.path.exists(IMAGE_METADATA):
        metadata_df = pd.read_csv(IMAGE_METADATA)
        print(f"Loaded metadata for {len(metadata_df)} images")
    
    results = []
    
    for idx, filename in enumerate(image_files):
        image_path = os.path.join(IMAGE_DIR, filename)
        
        print(f"Processing {idx + 1}/{len(image_files)}: {filename}...")
        
        # Get image info
        info = get_image_info(image_path)
        if info is None:
            continue
        
        # Get file size
        file_size = os.path.getsize(image_path)
        
        # Look up metadata
        post_id = os.path.splitext(filename)[0]
        meta_row = None
        if metadata_df is not None:
            matching = metadata_df[metadata_df['post_id'] == post_id]
            if not matching.empty:
                meta_row = matching.iloc[0]
        
        result = {
            "filename": filename,
            "post_id": post_id,
            "width": info["width"],
            "height": info["height"],
            "format": info["format"],
            "mode": info["mode"],
            "aspect_ratio": info["aspect_ratio"],
            "file_size_bytes": file_size,
            "file_size_kb": round(file_size / 1024, 2),
            "processed_at": datetime.now().isoformat()
        }
        
        # Add metadata if available
        if meta_row is not None:
            result["title"] = meta_row.get("title", "")
            result["subreddit"] = meta_row.get("subreddit", "")
            result["source_url"] = meta_row.get("source_url", "")
        
        results.append(result)
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved {len(results)} image analysis results to {OUTPUT_CSV}")
        
        # Print summary
        print("\n=== Image Summary ===")
        print(f"Total images: {len(results)}")
        print(f"Average resolution: {results_df['width'].mean():.0f} x {results_df['height'].mean():.0f}")
        print(f"Total size: {results_df['file_size_kb'].sum() / 1024:.2f} MB")
    else:
        print("No images processed")

if __name__ == "__main__":
    print("=== Starting Image Processing ===")
    process_images()
    print("=== Image Processing Complete ===")