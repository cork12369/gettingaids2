"""
STAGE 1: Data Scraper (keyless)
Replaces 01_scrape_reddit.py + 02_scrape_images.py

Text:   DuckDuckGo search snippets + blog scraping -> sentiment corpus
Images: DuckDuckGo image search + Wikimedia Commons -> visual corpus

Install: pip install duckduckgo-search requests beautifulsoup4 pillow pandas
No API keys required.
"""

import time
import random
import requests
import pandas as pd
from pathlib import Path
from io import BytesIO
from PIL import Image
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# -- Config -------------------------------------------------------------------

DATA_DIR     = Path("/data")
IMG_DIR      = Path("/data/images")
TEXT_CSV     = Path("/data/text_raw.csv")
IMG_META_CSV = Path("/data/image_metadata.csv")

MIN_IMG_SIZE   = 200    # px — skip thumbnails
MAX_TEXT_ITEMS = 80     # per query
MAX_IMG_ITEMS  = 60     # per query
SCRAPE_DELAY   = (1, 3) # random sleep range (seconds) — be polite

# Countries × search angles
# Text queries cast wide: reviews, opinions, travel blog language
TEXT_QUERIES = {
    "japan": [
        "manhole cover japan beautiful design",
        "japanese manhole cover tourism opinion",
        "decorative manhole cover japan travel review",
        "japan manhole art street photography",
        "pokemon manhole cover japan tourist",
    ],
    "singapore": [
        "manhole cover singapore street",
        "drain cover singapore design opinion",
        "singapore urban infrastructure review",
    ],
    "uk": [
        "manhole cover london design",
        "drain cover uk street opinion",
        "british manhole cover history",
    ],
    "usa": [
        "manhole cover new york design",
        "sewer cover usa street art",
        "american manhole cover opinion",
    ],
    "germany": [
        "manhole cover germany design kanaldeckel",
        "german drain cover street review",
    ],
    "france": [
        "manhole cover paris france design",
        "regard fonte paris street opinion",
    ],
    "india": [
        "manhole cover india design street",
        "drain cover mumbai infrastructure opinion",
    ],
}

# Image queries — more specific to get actual cover photos not street scenes
IMG_QUERIES = {
    "japan":     ["japanese manhole cover art design", "decorative manhole japan closeup", "pokemon manhole cover japan"],
    "singapore": ["singapore manhole cover", "drain cover singapore closeup"],
    "uk":        ["london manhole cover closeup", "british drain cover design"],
    "usa":       ["new york manhole cover closeup", "american sewer cover design"],
    "germany":   ["kanaldeckel design germany", "german manhole cover closeup"],
    "france":    ["regard fonte paris closeup", "french manhole cover design"],
    "india":     ["india manhole cover street", "mumbai drain cover closeup"],
}

# Wikimedia Commons categories — highest quality source, already labeled
WIKIMEDIA_CATEGORIES = {
    "japan":     "Manhole_covers_in_Japan",
    "singapore": "Manholes_in_Singapore",
    "uk":        "Manhole_covers_in_the_United_Kingdom",
    "usa":       "Manhole_covers_in_the_United_States",
    "germany":   "Kanaldeckel_in_Deutschland",
    "france":    "Manholes_in_France",
}

# Country keyword map for tagging scraped text by region
COUNTRY_KEYWORDS = {
    "japan":     ["japan", "japanese", "tokyo", "osaka", "kyoto", "nippon",
                  "manhole card", "pokemon lid", "pokefuta", "マンホール"],
    "singapore": ["singapore", " sg ", "singaporean"],
    "uk":        ["london", " uk ", "united kingdom", "britain", "england"],
    "usa":       ["new york", "nyc", "usa", "america", "american"],
    "germany":   ["germany", "german", "berlin", "kanaldeckel", "deutschland"],
    "france":    ["france", "paris", "french"],
    "india":     ["india", "indian", "mumbai", "delhi"],
}

# -- Helpers -----------------------------------------------------------------

def sleep():
    time.sleep(random.uniform(*SCRAPE_DELAY))


def infer_country(text: str) -> str:
    text = text.lower()
    for country, keywords in COUNTRY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return country
    return "unknown"


def safe_get(url: str, timeout: int = 12) -> requests.Response | None:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; research-scraper/1.0)"}
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        return resp
    except Exception as e:
        print(f"    ! GET failed {url[:60]}: {e}")
        return None


# -- Text Scraping -----------------------------------------------------------

def scrape_ddg_text(country: str, queries: list[str]) -> list[dict]:
    """
    Step 1: DDG gives us titles + snippets + URLs.
    Step 2: For promising URLs, fetch the actual page and extract body text.
    This gives much richer sentiment signal than snippets alone.
    """
    results = []

    with DDGS() as ddgs:
        for query in queries:
            print(f"  DDG text: '{query}'")
            try:
                hits = ddgs.text(query, max_results=MAX_TEXT_ITEMS)
            except Exception as e:
                print(f"    ! DDG error: {e}")
                sleep()
                continue

            for hit in hits:
                # Always save the snippet — it's guaranteed text
                snippet_text = f"{hit.get('title','')} {hit.get('body','')}".strip()
                results.append({
                    "country":  country,
                    "query":    query,
                    "source":   "ddg_snippet",
                    "url":      hit.get("href", ""),
                    "text":     snippet_text,
                    "score":    1,  # no upvote signal, weight equally
                })

                # Try to fetch full page for higher-value sources
                url = hit.get("href", "")
                if any(domain in url for domain in [
                    "tripadvisor", "reddit.com", "atlasobscura",
                    "timeout", "timeout", "lonelyplanet", "blog",
                    "travel", "japan", "substack",
                ]):
                    full_text = scrape_page_text(url)
                    if full_text:
                        results.append({
                            "country":  country,
                            "query":    query,
                            "source":   "full_page",
                            "url":      url,
                            "text":     full_text,
                            "score":    3,  # weight full pages higher
                        })

            sleep()

    return results


def scrape_page_text(url: str, max_chars: int = 3000) -> str | None:
    """Fetch a page and extract main body text, stripped of nav/footer noise."""
    resp = safe_get(url)
    if not resp:
        return None

    try:
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove boilerplate tags
        for tag in soup(["nav", "footer", "header", "script",
                          "style", "aside", "form", "iframe"]):
            tag.decompose()

        # Prefer article/main content blocks
        content = (
            soup.find("article") or
            soup.find("main") or
            soup.find(class_=lambda c: c and any(
                kw in c.lower() for kw in ["content", "article", "post", "body"]
            )) or
            soup.find("body")
        )

        if not content:
            return None

        text = " ".join(content.get_text(separator=" ").split())
        return text[:max_chars] if len(text) > 100 else None

    except Exception as e:
        print(f"    ! Parse error {url[:50]}: {e}")
        return None


# -- Image Scraping ----------------------------------------------------------

def scrape_ddg_images(country: str, queries: list[str]) -> list[dict]:
    """Fetch image URLs via DDG image search."""
    results = []

    with DDGS() as ddgs:
        for query in queries:
            print(f"  DDG images: '{query}'")
            try:
                hits = ddgs.images(query, max_results=MAX_IMG_ITEMS)
            except Exception as e:
                print(f"    ! DDG error: {e}")
                sleep()
                continue

            for hit in hits:
                url = hit.get("image", "")
                if url:
                    results.append({
                        "country": country,
                        "query":   query,
                        "source":  "ddg_image",
                        "url":     url,
                        "title":   hit.get("title", ""),
                    })

            sleep()

    return results


def scrape_wikimedia(country: str, category: str, limit: int = 100) -> list[dict]:
    """Fetch image URLs from a Wikimedia Commons category."""
    results = []
    api_url = "https://commons.wikimedia.org/w/api.php"

    params = {
        "action":  "query",
        "list":    "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype":  "file",
        "cmlimit": min(limit, 500),
        "format":  "json",
    }

    resp = safe_get(api_url + "?" + "&".join(f"{k}={v}" for k, v in params.items()))
    if not resp:
        return results

    members = resp.json().get("query", {}).get("categorymembers", [])
    print(f"  Wikimedia [{country}]: {len(members)} files in {category}")

    for m in members:
        img_url = resolve_wikimedia_url(m["title"])
        if img_url:
            results.append({
                "country": country,
                "query":   category,
                "source":  "wikimedia",
                "url":     img_url,
                "title":   m["title"],
            })
        time.sleep(0.15)

    return results


def resolve_wikimedia_url(title: str) -> str | None:
    """Resolve a Wikimedia file title to its direct image URL."""
    api_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action":    "query",
        "titles":    title,
        "prop":      "imageinfo",
        "iiprop":    "url",
        "iiurlwidth": 800,
        "format":    "json",
    }
    try:
        resp = requests.get(api_url, params=params, timeout=10)
        pages = resp.json()["query"]["pages"]
        page = next(iter(pages.values()))
        return page["imageinfo"][0]["thumburl"]
    except Exception:
        return None


# -- Image Downloader --------------------------------------------------------

def download_images(metadata_list: list[dict]) -> list[dict]:
    """Download and validate images, save to /data/images/<country>/."""
    successful = []
    seen_urls = set()

    for item in metadata_list:
        url = item.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        country_dir = IMG_DIR / item["country"]
        country_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize filename
        safe_id = url.split("/")[-1][:40].replace("%", "_").replace(" ", "_")
        filename = f"{item['source']}_{safe_id}.jpg"
        filepath = country_dir / filename

        if filepath.exists():
            item["local_path"] = str(filepath)
            successful.append(item)
            continue

        resp = safe_get(url, timeout=15)
        if not resp:
            continue

        try:
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            if min(img.size) < MIN_IMG_SIZE:
                continue

            img.save(filepath, "JPEG", quality=90)
            item["local_path"] = str(filepath)
            item["width"]      = img.size[0]
            item["height"]     = img.size[1]
            successful.append(item)

        except Exception as e:
            print(f"    ! Image save failed: {e}")

        time.sleep(0.3)

    return successful


# -- Main --------------------------------------------------------------------

def run():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    all_text   = []
    all_images = []

    # -- Text scraping --------------------------------------------------------
    print("\n=== TEXT SCRAPING ===")
    for country, queries in TEXT_QUERIES.items():
        print(f"\n[{country.upper()}]")
        items = scrape_ddg_text(country, queries)
        all_text.extend(items)
        print(f"  -> {len(items)} text records")

    # Save text corpus
    text_df = pd.DataFrame(all_text)
    text_df = text_df[text_df["text"].str.strip().str.len() > 30]
    text_df = text_df.drop_duplicates(subset=["url", "source"])
    text_df.to_csv(TEXT_CSV, index=False)
    print(f"\nText corpus: {len(text_df)} records -> {TEXT_CSV}")
    print(text_df.groupby(["country", "source"]).size().to_string())

    # -- Image scraping --------------------------------------------------------
    print("\n\n=== IMAGE SCRAPING ===")

    # DDG images
    for country, queries in IMG_QUERIES.items():
        print(f"\n[{country.upper()}] DDG")
        items = scrape_ddg_images(country, queries)
        all_images.extend(items)
        print(f"  -> {len(items)} image URLs")

    # Wikimedia (higher quality, keep separate)
    print("\n[WIKIMEDIA]")
    for country, category in WIKIMEDIA_CATEGORIES.items():
        items = scrape_wikimedia(country, category)
        all_images.extend(items)

    # Download
    print(f"\nDownloading {len(all_images)} images...")
    downloaded = download_images(all_images)

    img_df = pd.DataFrame(downloaded)
    img_df.to_csv(IMG_META_CSV, index=False)
    print(f"\nImages: {len(downloaded)} downloaded -> {IMG_META_CSV}")
    print(img_df.groupby(["country", "source"]).size().to_string())

    print("\n=== SCRAPING COMPLETE ===")
    print(f"  Text records : {len(text_df)}")
    print(f"  Images       : {len(downloaded)}")
    print(f"\nNext: run 02_sentiment_analysis.py")


if __name__ == "__main__":
    run()
