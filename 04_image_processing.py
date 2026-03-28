"""
STAGE 3: Image Processing
Process and analyze manhole cover images scraped by 01_scrape_data.py.

Reads from:
  /data/images/<country>/  (organized by country)
  /data/image_metadata.csv (from Stage 1)

Outputs:
  /data/output/image_analysis.csv
"""

import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
IMAGE_DIR       = Path("/data/images")
IMAGE_METADATA  = Path("/data/image_metadata.csv")  # Updated to match Stage 1 output
OUTPUT_DIR      = Path("/data/output")
OUTPUT_CSV      = OUTPUT_DIR / "image_analysis.csv"

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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if image directory exists
    if not IMAGE_DIR.exists():
        print(f"ERROR: {IMAGE_DIR} not found. Run 01_scrape_data.py first.")
        return
    
    # Get all image files recursively (from country subdirectories)
    image_files = []
    for ext in SUPPORTED_FORMATS:
        image_files.extend(IMAGE_DIR.rglob(f"*{ext}"))
        image_files.extend(IMAGE_DIR.rglob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images in {IMAGE_DIR}")
    
    # Load metadata if available
    metadata_df = None
    if IMAGE_METADATA.exists():
        metadata_df = pd.read_csv(IMAGE_METADATA)
        print(f"Loaded metadata for {len(metadata_df)} images")
    
    results = []
    
    for idx, image_path in enumerate(image_files):
        filename = image_path.name
        relative_path = str(image_path.relative_to(IMAGE_DIR))
        
        print(f"Processing {idx + 1}/{len(image_files)}: {relative_path}...")
        
        # Get image info
        info = get_image_info(image_path)
        if info is None:
            continue
        
        # Get file size
        file_size = os.path.getsize(image_path)
        
        # Extract country from path if organized by country
        parts = relative_path.split(os.sep)
        country = parts[0] if len(parts) > 1 else "unknown"
        
        # Look up metadata by local_path or filename
        meta_row = None
        if metadata_df is not None:
            # Try matching by local_path first
            if "local_path" in metadata_df.columns:
                matching = metadata_df[metadata_df["local_path"].str.contains(filename, na=False, regex=False)]
            else:
                matching = pd.DataFrame()
            
            if matching.empty:
                # Fallback: match by filename in url or title
                matching = metadata_df[
                    metadata_df.get("url", pd.Series()).str.contains(filename[:30], na=False, regex=False)
                ]
            
            if not matching.empty:
                meta_row = matching.iloc[0]
        
        result = {
            "filename": filename,
            "relative_path": relative_path,
            "country": country,
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
            result["source"] = meta_row.get("source", "")
            result["source_url"] = meta_row.get("url", "")
        
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
        
        # By country
        if "country" in results_df.columns:
            print("\nBy country:")
            print(results_df.groupby("country").size().to_string())
    else:
        print("No images processed")


if __name__ == "__main__":
    print("=== Starting Image Processing ===")
    process_images()
    print("=== Image Processing Complete ===")