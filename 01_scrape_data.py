"""
STAGE 1: Data Scraper (keyless)
Replaces 01_scrape_reddit.py + 02_scrape_images.py

Text:   DuckDuckGo search snippets + blog scraping → sentiment corpus
Images: DuckDuckGo image search + Wikimedia Commons + Fallback sources → visual corpus

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
from collections import defaultdict
import functools
import logging

# ── Logging Setup ───────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ── Retry Decorator ─────────────────────────────────────────────────────────────
def retry_with_backoff(max_retries=3, base_delay=1, backoff_factor=2):
    """Decorator that retries a function with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = base_delay
            last_exception = None
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    retries += 1
                    if retries < max_retries:
                        delay *= backoff_factor
                        logger.warning(f"Retry {retries}/{max_retries} for {func.__name__}: {e}")
                        time.sleep(delay)
            if last_exception:
                raise last_exception
            return None
        return wrapper
    return decorator


# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR     = Path("/data")
IMG_DIR      = Path("/data/images")
TEXT_CSV     = Path("/data/text_raw.csv")
IMG_META_CSV = Path("/data/image_metadata.csv")

MIN_IMG_SIZE      = 200    # px — skip thumbnails
MAX_TEXT_ITEMS    = 80     # per query
MAX_IMG_ITEMS     = 60     # per query
SCRAPE_DELAY      = (1, 3) # random sleep range (seconds) — be polite

# Balancing config
MIN_TEXT_PER_COUNTRY = 10   # target minimum text records per country
MAX_TEXT_PER_COUNTRY = 100  # cap to prevent over-representation
MIN_IMG_PER_COUNTRY  = 15   # target minimum images per country
MAX_IMG_PER_COUNTRY  = 80   # cap to prevent over-representation

# ── Target Image Count ───────────────────────────────────────────────────────────
# This is the target we try to hit for all countries
TARGET_IMG_PER_COUNTRY = 50  # ideal number of images per country

# Timeout config (seconds)
REQUEST_TIMEOUT = 15
DOWNLOAD_TIMEOUT = 20
CONNECT_TIMEOUT = 10

# Countries × search angles
# Text queries cast wide: reviews, opinions, travel blog language
TEXT_QUERIES = {
    "japan": [
        "manhole cover japan beautiful design",
        "japanese manhole cover tourism opinion",
        "decorative manhole cover japan travel review",
        "japan manhole art street photography",
        "pokemon manhole cover japan tourist",
        "pokefuta japan design",
        "マンホール カード 日本",
    ],
    "singapore": [
        "manhole cover singapore street",
        "drain cover singapore design opinion",
        "singapore urban infrastructure review",
        "singapore utility cover design",
    ],
    "uk": [
        "manhole cover london design",
        "drain cover uk street opinion",
        "british manhole cover history",
        "uk utility cover art",
        "london street cover design",
    ],
    "usa": [
        "manhole cover new york design",
        "sewer cover usa street art",
        "american manhole cover opinion",
        "us utility cover design review",
        "san francisco manhole art",
    ],
    "germany": [
        "manhole cover germany design kanaldeckel",
        "german drain cover street review",
        "kanaldeckel kunst deutschland",
        "berlin manhole cover design",
    ],
    "france": [
        "manhole cover paris france design",
        "regard fonte paris street opinion",
        "french utility cover art",
        "égout paris design",
    ],
    "india": [
        "manhole cover india design street",
        "drain cover mumbai infrastructure opinion",
        "indian utility cover art",
        "delhi manhole design",
    ],
    "italy": [
        "manhole cover italy design",
        "italian drain cover rome",
        "chiusino milano design",
        "italy street art utility cover",
    ],
    "spain": [
        "manhole cover spain design",
        "spanish drain cover madrid",
        "tapas registro barcelona",
        "spain utility cover art",
    ],
    "australia": [
        "manhole cover australia design",
        "australian drain cover sydney",
        "melbourne utility cover",
    ],
    "canada": [
        "manhole cover toronto design",
        "canadian drain cover vancouver",
        "canada utility cover art",
    ],
    "brazil": [
        "manhole cover brazil design",
        "brazilian drain cover sao paulo",
        "boca de lobo rio design",
    ],
    "netherlands": [
        "manhole cover netherlands design",
        "dutch drain cover amsterdam",
        "riooldeksel nederland",
    ],
    "south_korea": [
        "manhole cover south korea design",
        "korean drain cover seoul",
        "seoul utility cover art",
    ],
    "thailand": [
        "manhole cover thailand design",
        "thai drain cover bangkok",
        "thailand utility cover art",
    ],
    "mexico": [
        "manhole cover mexico design",
        "mexican drain cover cdmx",
        "alcantarilla mexico city art",
    ],
}

# Image queries — more specific to get actual cover photos not street scenes
IMG_QUERIES = {
    "japan":     ["japanese manhole cover art design", "decorative manhole japan closeup", "pokemon manhole cover japan", "pokefuta japan"],
    "singapore": ["singapore manhole cover", "drain cover singapore closeup", "singapore utility cover"],
    "uk":        ["london manhole cover closeup", "british drain cover design", "uk utility cover"],
    "usa":       ["new york manhole cover closeup", "american sewer cover design", "usa utility cover art"],
    "germany":   ["kanaldeckel design germany", "german manhole cover closeup", "deutschland kanaldeckel kunst"],
    "france":    ["regard fonte paris closeup", "french manhole cover design", "france utility cover"],
    "india":     ["india manhole cover street", "mumbai drain cover closeup", "indian utility cover"],
    "italy":     ["italian manhole cover design", "roma chiusino", "italy drain cover"],
    "spain":     ["spanish manhole cover", "madrid tapas registro", "spain drain cover"],
    "australia": ["australia manhole cover", "sydney drain cover", "melbourne utility cover"],
    "canada":    ["canada manhole cover", "toronto drain cover", "vancouver utility cover"],
    "brazil":    ["brazil manhole cover", "sao paulo drain cover", "rio boca de lobo"],
    "netherlands": ["netherlands manhole cover", "amsterdam drain cover", "dutch utility cover"],
    "south_korea": ["korea manhole cover", "seoul drain cover", "korean utility cover art"],
    "thailand":  ["thailand manhole cover", "bangkok drain cover", "thai utility cover"],
    "mexico":    ["mexico manhole cover", "cdmx drain cover", "mexican utility cover art"],
}

# Wikimedia Commons categories — highest quality source, already labeled
WIKIMEDIA_CATEGORIES = {
    "japan":     "Manhole_covers_in_Japan",
    "singapore": "Manholes_in_Singapore",
    "uk":        "Manhole_covers_in_the_United_Kingdom",
    "usa":       "Manhole_covers_in_the_United_States",
    "germany":   "Kanaldeckel_in_Deutschland",
    "france":    "Manholes_in_France",
    "italy":     "Manhole_covers_in_Italy",
    "netherlands": "Gully_covers_in_the_Netherlands",
    "spain":     "Manhole_covers_in_Spain",
    "australia": "Manhole_covers_in_Australia",
}

# ── Fallback Image Sources ────────────────────────────────────────────────────────────
# When primary sources fail, these provide additional image URLs
FALLBACK_SOURCES = {
    "unsplash": {
        "url": "https://api.unsplash.com/photos/random",
        "params": {"query": "manhole cover", "count": 30},
    },
    "flickr": {
        "search_url": "https://www.flickr.com/services/rest/",
        "params": {
            "method": "flickr.photos.search",
            "api_key": "none",
            "text": "manhole cover",
            "per_page": 50,
            "format": "json",
        },
    },
    "pixabay": {
        "url": "https://pixabay.com/api/",
        "params": {"q": "manhole cover", "image_type": "photo", "per_page": 50},
    },
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
    "italy":     ["italy", "italian", "rome", "milan", "roma", "milano"],
    "spain":     ["spain", "spanish", "madrid", "barcelona"],
    "australia": ["australia", "australian", "sydney", "melbourne"],
    "canada":    ["canada", "canadian", "toronto", "vancouver"],
    "brazil":    ["brazil", "brazilian", "sao paulo", "rio"],
    "netherlands": ["netherlands", "dutch", "amsterdam", "holland"],
    "south_korea": ["korea", "korean", "seoul", "south korea"],
    "thailand":  ["thailand", "thai", "bangkok"],
    "mexico":    ["mexico", "mexican", "cdmx", "mexico city"],
}


# ── HTTP Helpers with Error Handling ───────────────────────────────────────

def safe_get(url: str, timeout: int = REQUEST_TIMEOUT) -> requests.Response | None:
    """Make HTTP GET request with retry and timeout handling."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-scraper/1.0)"}
    try:
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        return resp
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout ({timeout}s) for {url[:60]}")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error for {url[:60]}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {url[:60]}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error for {url[:60]}: {e}")
        return None


def safe_post(url: str, data: dict = None, timeout: int = REQUEST_TIMEOUT) -> requests.Response | None:
    """Make HTTP POST request with retry and timeout handling."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-scraper/1.0)"}
    try:
        resp = requests.post(url, json=data, timeout=timeout, headers=headers)
        resp.raise_for_status()
        return resp
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout ({timeout}s) for {url[:60]}")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error for {url[:60]}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {url[:60]}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error for {url[:60]}: {e}")
        return None


def sleep():
    time.sleep(random.uniform(*SCRAPE_DELAY))


def infer_country(text: str) -> str:
    text = text.lower()
    for country, keywords in COUNTRY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return country
    return "unknown"


# ── Fallback Image Sources ───────────────────────────────────────────────────────────

def fetch_unsplash_images(country: str, count: int = 30) -> list[dict]:
    """Fetch images from Unsplash as fallback source."""
    try:
        resp = safe_get(FALLBACK_SOURCES["unsplash"]["url"])
        if not resp:
            return []
        
        data = resp.json()
        # Handle single response or list
        if isinstance(data, dict):
            urls = [data.get("urls", {}).get("regular", "")]
        else:
            urls = [item.get("urls", {}).get("regular", "") for item in data]
        
        results = []
        for url in urls[:count]:
            if url:
                results.append({
                    "country": country,
                    "query":   "unsplash_fallback",
                    "source":  "unsplash",
                    "url":     url,
                    "title":   "Unsplash fallback image",
                })
        return results
    except Exception as e:
        logger.error(f"Unsplash fallback failed for {country}: {e}")
        return []


def fetch_flickr_images(country: str, count: int = 30) -> list[dict]:
    """Fetch images from Flickr as fallback source (no API key required for free tier)."""
    try:
        params = FALLBACK_SOURCES["flickr"]["params"].copy()
        params["text"] = f"manhole cover {country}"
        resp = safe_get(FALLBACK_SOURCES["flickr"]["search_url"], timeout=REQUEST_TIMEOUT)
        if not resp:
            return []
        
        data = resp.json()
        photos = data.get("photos", {}).get("photo", [])
        
        results = []
        for photo in photos[:count]:
            url = f"https://live.staticflickr.com/{photo['server']}/{photo['id']}_{photo['secret']}.jpg"
            results.append({
                "country": country,
                "query":   "flickr_fallback",
                "source":  "flickr",
                "url":     url,
                "title":   photo.get("title", ""),
            })
        return results
    except Exception as e:
        logger.error(f"Flickr fallback failed for {country}: {e}")
        return []


def fetch_pixabay_images(country: str, count: int = 30) -> list[dict]:
    """Fetch images from Pixabay as fallback source."""
    try:
        resp = safe_get(FALLBACK_SOURCES["pixabay"]["url"])
        if not resp:
            return []
        
        data = resp.json()
        hits = data.get("hits", [])
        
        results = []
        for hit in hits[:count]:
            url = hit.get("webformatURL", "")
            if url:
                results.append({
                    "country": country,
                    "query":   "pixabay_fallback",
                    "source":  "pixabay",
                    "url":     url,
                    "title":   hit.get("tags", ""),
                })
        return results
    except Exception as e:
        logger.error(f"Pixabay fallback failed for {country}: {e}")
        return []


def fetch_fallback_images(country: str, needed_count: int) -> list[dict]:
    """Try multiple fallback sources to get needed images for a country."""
    all_results = []
    
    # Try Unsplash first
    results = fetch_unsplash_images(country, needed_count)
    all_results.extend(results)
    logger.info(f"  Unsplash: {len(results)} images for {country}")
    
    # If still need more, try Flickr
    if len(all_results) < needed_count:
        remaining = needed_count - len(all_results)
        results = fetch_flickr_images(country, remaining)
        all_results.extend(results)
        logger.info(f"  Flickr: {len(results)} images for {country}")
    
    # If still need more, try Pixabay
    if len(all_results) < needed_count:
        remaining = needed_count - len(all_results)
        results = fetch_pixabay_images(country, remaining)
        all_results.extend(results)
        logger.info(f"  Pixabay: {len(results)} images for {country}")
    
    return all_results


# ── Balancing Functions ───────────────────────────────────────────────────────────────────

def balance_text_records(records: list[dict], min_per_country: int = MIN_TEXT_PER_COUNTRY, 
                         max_per_country: int = MAX_TEXT_PER_COUNTRY) -> list[dict]:
    """
    Balance text records across countries to ensure minimum representation
    and cap over-represented countries.
    """
    # Group by country
    by_country = defaultdict(list)
    for r in records:
        by_country[r["country"]].append(r)
    
    balanced = []
    countries_meeting_min = 0
    
    for country, items in by_country.items():
        if country == "unknown":
            continue
        
        # Deduplicate by URL
        seen_urls = set()
        unique_items = []
        for item in items:
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_items.append(item)
        
        # Sort by score (higher quality first)
        unique_items.sort(key=lambda x: x.get("score", 1), reverse=True)
        
        # Take up to max_per_country
        selected = unique_items[:max_per_country]
        balanced.extend(selected)
        
        if len(selected) >= min_per_country:
            countries_meeting_min += 1
        else:
            logger.warning(f"{country}: only {len(selected)} records (target: {min_per_country})")
    
    logger.info(f"Text balance: {countries_meeting_min}/{len(by_country)} countries meet min {min_per_country}")
    return balanced


def balance_image_candidates(candidates: list[dict], min_per_country: int = MIN_IMG_PER_COUNTRY,
                             max_per_country: int = MAX_IMG_PER_COUNTRY) -> list[dict]:
    """
    Balance image candidates across countries to ensure broad coverage.
    Fetch from fallback sources if a country is under target.
    """
    # Group by country
    by_country = defaultdict(list)
    for c in candidates:
        by_country[c["country"]].append(c)
    
    balanced = []
    countries_with_images = 0
    
    for country, items in by_country.items():
        if country == "unknown":
            continue
        
        # Deduplicate by URL
        seen_urls = set()
        unique_items = []
        for item in items:
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_items.append(item)
        
        # Prioritize wikimedia (higher quality)
        unique_items.sort(key=lambda x: (x.get("source") == "wikimedia",), reverse=True)
        
        # Check if we need fallback images
        if len(unique_items) < min_per_country:
            needed = min_per_country - len(unique_items)
            logger.info(f"Fetching {needed} fallback images for {country}")
            fallback = fetch_fallback_images(country, needed)
            unique_items.extend(fallback)
        
        # Take up to max_per_country
        selected = unique_items[:max_per_country]
        balanced.extend(selected)
        
        if len(selected) >= min_per_country:
            countries_with_images += 1
        else:
            logger.warning(f"{country}: only {len(selected)} image URLs (target: {min_per_country})")
    
    logger.info(f"Image balance: {countries_with_images}/{len(by_country)} countries meet min {min_per_country}")
    return balanced


# ── Text Scraping ─────────────────────────────────────────────────────────────────

def scrape_ddg_text(country: str, queries: list[str]) -> list[dict]:
    """
    Step 1: DDG gives us titles + snippets + URLs.
    Step 2: For promising URLs, fetch the actual page and extract body text.
    This gives much richer sentiment signal than snippets alone.
    """
    results = []

    with DDGS() as ddgs:
        for query in queries:
            logger.info(f"DDG text: '{query}'")
            try:
                hits = ddgs.text(query, max_results=MAX_TEXT_ITEMS)
            except Exception as e:
                logger.error(f"DDG error: {e}")
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
                    "timeout", "lonelyplanet", "blog",
                    "travel", "japan", "substack", "medium", "wordpress",
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
        logger.error(f"Parse error {url[:50]}: {e}")
        return None


# ── Image Scraping ────────────────────────────────────────────────────────────

def scrape_ddg_images(country: str, queries: list[str]) -> list[dict]:
    """Fetch image URLs via DDG image search."""
    results = []

    with DDGS() as ddgs:
        for query in queries:
            logger.info(f"DDG images: '{query}'")
            try:
                hits = ddgs.images(query, max_results=MAX_IMG_ITEMS)
            except Exception as e:
                logger.error(f"DDG error: {e}")
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
    
    try:
        members = resp.json().get("query", {}).get("categorymembers", [])
        logger.info(f"Wikimedia [{country}]: {len(members)} files in {category}")
        
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
    except Exception as e:
        logger.error(f"Wikimedia parse error for {country}: {e}")

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
        resp = requests.get(api_url, params=params, timeout=CONNECT_TIMEOUT)
        pages = resp.json()["query"]["pages"]
        page = next(iter(pages.values()))
        return page["imageinfo"][0]["thumburl"]
    except Exception as e:
        logger.error(f"Wikimedia URL resolution failed: {e}")
        return None


# ── Image Downloader ──────────────────────────────────────────────────────────

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

        resp = safe_get(url, timeout=DOWNLOAD_TIMEOUT)
        if not resp:
            continue

        try:
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            if min(img.size) < MIN_IMG_SIZE:
                logger.warning(f"Skipping thumbnail: {min(img.size)}px < {MIN_IMG_SIZE}px")
                continue

            img.save(filepath, "JPEG", quality=90)
            item["local_path"] = str(filepath)
            item["width"]      = img.size[0]
            item["height"]     = img.size[1]
            successful.append(item)

        except Exception as e:
            logger.error(f"Image save failed: {e}")

        time.sleep(0.3)

    return successful


# ── Main ───────────────────────────────────────────────────────────────────────────────

def run():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    all_text   = []
    all_images = []

    # ── Text scraping ──────────────────────────────────────────────────────────
    logger.info("=== TEXT SCRAPING ===")
    for country, queries in TEXT_QUERIES.items():
        logger.info(f"[{country.upper()}]")
        items = scrape_ddg_text(country, queries)
        all_text.extend(items)
        logger.info(f"  → {len(items)} text records")

    # Balance text records
    logger.info("=== BALANCING TEXT RECORDS ===")
    all_text = balance_text_records(all_text)

    # Save text corpus
    text_df = pd.DataFrame(all_text)
    text_df = text_df[text_df["text"].str.strip().str.len() > 30]
    text_df = text_df.drop_duplicates(subset=["url", "source"])
    text_df.to_csv(TEXT_CSV, index=False)
    logger.info(f"Text corpus: {len(text_df)} records → {TEXT_CSV}")
    
    # Print country distribution
    logger.info("=== TEXT BY COUNTRY ===")
    country_counts = text_df.groupby("country").size().sort_values(ascending=False)
    logger.info(country_counts.to_string())
    logger.info(f"Countries with ≥{MIN_TEXT_PER_COUNTRY} records: {(country_counts >= MIN_TEXT_PER_COUNTRY).sum()}")

    # ── Image scraping ─────────────────────────────────────────────────────────
    logger.info("=== IMAGE SCRAPING ===")

    # DDG images
    for country, queries in IMG_QUERIES.items():
        logger.info(f"[{country.upper()}] DDG")
        items = scrape_ddg_images(country, queries)
        all_images.extend(items)
        logger.info(f"  → {len(items)} image URLs")

    # Wikimedia (higher quality, keep separate)
    logger.info("[WIKIMEDIA]")
    for country, category in WIKIMEDIA_CATEGORIES.items():
        items = scrape_wikimedia(country, category)
        all_images.extend(items)
        logger.info(f"  → {len(items)} Wikimedia images for {country}")

    # Balance image candidates (with fallback sources)
    logger.info("=== BALANCING IMAGE CANDIDATES ===")
    all_images = balance_image_candidates(all_images)

    # Download
    logger.info(f"Downloading {len(all_images)} balanced images...")
    downloaded = download_images(all_images)

    img_df = pd.DataFrame(downloaded)
    img_df.to_csv(IMG_META_CSV, index=False)
    logger.info(f"Images: {len(downloaded)} downloaded → {IMG_META_CSV}")
    
    # Print image distribution
    logger.info("=== IMAGES BY COUNTRY ===")
    img_country_counts = img_df.groupby("country").size().sort_values(ascending=False)
    logger.info(img_country_counts.to_string())
    logger.info(f"Countries with ≥{MIN_IMG_PER_COUNTRY} images: {(img_country_counts >= MIN_IMG_PER_COUNTRY).sum()}")

    
    # Print balance summary
    logger.info("=== SCRAPING COMPLETE ===")
    logger.info(f"  Text records : {len(text_df)}")
    logger.info(f"  Images       : {len(downloaded)}")
    logger.info(f"  Countries    : {len(country_counts)} (text), {len(img_country_counts)} (images)")
    logger.info("Next: run 03_sentiment_analysis.py")


if __name__ == "__main__":
    run()