"""
STAGE 1: Data Scraper (keyless)
Replaces 01_scrape_reddit.py + 02_scrape_images.py

Text:   DuckDuckGo search snippets + blog scraping -> sentiment corpus
Images: DuckDuckGo image search + Wikimedia Commons -> visual corpus

Install: pip install ddgs requests beautifulsoup4 pillow pandas
No API keys required.
"""

import time
import random
import re
import os
import requests
import pandas as pd
from collections import Counter
from pathlib import Path
from io import BytesIO
from urllib.parse import quote_plus
from PIL import Image
from bs4 import BeautifulSoup
from ddgs import DDGS

# -- Config -------------------------------------------------------------------

DATA_DIR     = Path(os.getenv("DATA_DIR", "data"))
IMG_DIR      = DATA_DIR / "images"
TEXT_CSV     = DATA_DIR / "text_raw.csv"
IMG_META_CSV = DATA_DIR / "image_metadata.csv"

MIN_IMG_SIZE   = 200    # px — skip thumbnails
MAX_TEXT_ITEMS = 80     # per query
MAX_IMG_ITEMS  = 60     # per query
SCRAPE_DELAY   = (1, 3) # random sleep range (seconds) — be polite

# Data volume target — keep scraping until this many total data points are collected
MIN_TOTAL_DATA_POINTS = 2000
MAX_ROUNDS            = 5    # max retry rounds to reach the target

# Optional source toggles
ENABLE_MASTODON  = os.getenv("ENABLE_MASTODON", "true").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_PINTEREST = os.getenv("ENABLE_PINTEREST", "false").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_MAPILLARY = os.getenv("ENABLE_MAPILLARY", "false").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_YOUTUBE    = os.getenv("ENABLE_YOUTUBE", "true").strip().lower() in {"1", "true", "yes", "on"}
YOUTUBE_API_KEY   = os.getenv("YOUTUBE_API_KEY", "").strip()
YOUTUBE_MAX_RESULTS = int(os.getenv("YOUTUBE_MAX_RESULTS", "25"))  # per query

MASTODON_INSTANCES = [
    x.strip() for x in os.getenv(
        "MASTODON_INSTANCES",
        "mastodon.social,mastodon.world,fosstodon.org"
    ).split(",") if x.strip()
]

MAPILLARY_ACCESS_TOKEN = os.getenv("MAPILLARY_ACCESS_TOKEN", "").strip()

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

# YouTube search queries — video titles/descriptions for text, thumbnails for images
YOUTUBE_QUERIES = {
    "japan": [
        "japanese manhole cover art",
        "manhole cover japan tourism",
        "pokemon manhole cover japan pokefuta",
        "drainspotting japan documentary",
    ],
    "singapore": [
        "singapore manhole cover design",
        "singapore urban infrastructure",
    ],
    "uk": [
        "london manhole cover design",
        "british drain cover history",
    ],
    "usa": [
        "new york manhole cover design",
        "american sewer cover art",
    ],
    "germany": [
        "kanaldeckel germany manhole cover",
        "german manhole cover design",
    ],
    "france": [
        "paris manhole cover design regard",
        "french manhole cover art",
    ],
    "india": [
        "india manhole cover street",
        "mumbai drain cover infrastructure",
    ],
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

# ── Expanded queries for retry rounds ────────────────────────────────────────
# These are used in subsequent rounds when the initial scrape doesn't reach
# the MIN_TOTAL_DATA_POINTS target.  Each round picks the next list entry.

EXPANDED_TEXT_QUERIES = {
    "japan": [
        "manhole cover japan reddit",
        "manhole art japan blog",
        "drainspotting japan tokyo osaka",
        "japanese sewer cover beautiful art opinion",
        "pokefuta pokemon manhole japan review",
        "japan street infrastructure design opinion",
        "manhole cover japan culture tourism experience",
        "japan drain cover photo art appreciation",
    ],
    "singapore": [
        "manhole cover singapore reddit",
        "singapore drain cover blog review",
        "singapore street design infrastructure opinion",
        "drain cover singapore art beautiful",
        "singapore urban street furniture design",
    ],
    "uk": [
        "manhole cover uk reddit",
        "british drain cover blog review",
        "london sewer cover history opinion",
        "uk street drain cover photography",
        "manhole cover britain industrial design",
    ],
    "usa": [
        "manhole cover usa reddit",
        "american manhole cover blog opinion",
        "new york sewer cover art design",
        "us drain cover street photography review",
        "manhole cover usa history municipal",
    ],
    "germany": [
        "kanaldeckel deutschland blog meinung",
        "german manhole cover review opinion",
        "berlin drain cover design photography",
        "manhole cover germany art street",
    ],
    "france": [
        "manhole cover paris blog opinion",
        "regard fonte france street review",
        "french sewer cover design photography",
        "paris drain cover art history",
    ],
    "india": [
        "manhole cover india reddit opinion",
        "india drain cover blog review",
        "mumbai manhole street infrastructure",
        "indian sewer cover design municipal",
    ],
}

EXPANDED_IMG_QUERIES = {
    "japan":     ["japan manhole art photo", "pokefuta pokemon drain cover",
                  "tokyo osaka manhole design", "japanese sewer lid colorful"],
    "singapore": ["singapore drain cover photo", "singapore street utility cover"],
    "uk":        ["uk manhole cover photo", "british drain lid design vintage"],
    "usa":       ["usa manhole cover photo art", "american sewer lid design vintage"],
    "germany":   ["deutschland kanaldeckel foto", "german drain cover design photo"],
    "france":    ["paris regard fonte photo", "french manhole cover art"],
    "india":     ["india manhole cover photo", "mumbai drain cover street photo"],
}

# Approximate city-level bboxes for Mapillary image enrichment
# bbox format: min_lon,min_lat,max_lon,max_lat
MAPILLARY_BBOX = {
    "japan":     "139.65,35.62,139.84,35.74",      # Tokyo
    "singapore": "103.78,1.26,103.93,1.39",        # Singapore
    "uk":        "-0.24,51.47,-0.01,51.56",        # London
    "usa":       "-74.06,40.68,-73.90,40.83",      # NYC
    "germany":   "13.30,52.45,13.50,52.57",        # Berlin
    "france":    "2.25,48.82,2.42,48.90",          # Paris
    "india":     "72.77,18.89,72.99,19.07",        # Mumbai
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


def safe_get(url: str, timeout: int = 30) -> requests.Response | None:
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


def scrape_mastodon(country: str, queries: list[str], per_query: int = 20) -> tuple[list[dict], list[dict]]:
    """Best-effort Mastodon scraping from public instance search APIs."""
    text_results: list[dict] = []
    image_results: list[dict] = []
    seen_status_urls = set()
    seen_image_urls = set()

    for query in queries:
        print(f"  Mastodon: '{query}'")
        query_success = False

        for instance in MASTODON_INSTANCES:
            base = f"https://{instance}/api/v2/search"
            params = {
                "q": query,
                "type": "statuses",
                "limit": min(per_query, 40),
                "resolve": "true",
            }

            try:
                resp = requests.get(
                    base,
                    params=params,
                    timeout=20,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; research-scraper/1.0)"},
                )
                resp.raise_for_status()
                statuses = resp.json().get("statuses", [])
            except Exception as e:
                print(f"    ! Mastodon [{instance}] failed: {e}")
                continue

            print(f"    {instance}: {len(statuses)} statuses")
            query_success = True

            txt_added = 0
            img_added = 0
            for st in statuses:
                status_url = st.get("url") or st.get("uri") or ""

                # content is HTML on Mastodon
                raw_html = st.get("content", "")
                plain = BeautifulSoup(raw_html, "html.parser").get_text(" ", strip=True)
                if plain and len(plain) > 25 and status_url and status_url not in seen_status_urls:
                    text_results.append({
                        "country": country,
                        "query": query,
                        "source": "mastodon",
                        "url": status_url,
                        "text": plain,
                        "score": 1,
                    })
                    seen_status_urls.add(status_url)
                    txt_added += 1

                for media in st.get("media_attachments", []) or []:
                    if (media.get("type") or "").lower() != "image":
                        continue
                    img_url = media.get("url") or media.get("preview_url") or ""
                    if not img_url or img_url in seen_image_urls:
                        continue

                    image_results.append({
                        "country": country,
                        "query": query,
                        "source": "mastodon_image",
                        "url": img_url,
                        "title": media.get("description") or f"Mastodon image ({country})",
                    })
                    seen_image_urls.add(img_url)
                    img_added += 1

            print(f"    added text={txt_added}, images={img_added}")
            break

        if not query_success:
            print("    ! Mastodon: no instance returned usable results")
        sleep()

    return text_results, image_results


def scrape_pinterest(country: str, queries: list[str], max_hits_per_query: int = 60) -> tuple[list[dict], list[dict]]:
    """Best-effort Pinterest scraping from public search pages (may be fragile)."""
    text_results: list[dict] = []
    image_results: list[dict] = []
    seen_imgs = set()

    for query in queries:
        print(f"  Pinterest: '{query}'")
        url = f"https://www.pinterest.com/search/pins/?q={quote_plus(query)}"
        resp = safe_get(url, timeout=25)
        if not resp:
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        imgs = soup.find_all("img")
        if not imgs:
            print("    ! Pinterest returned no <img> tags (likely anti-bot or JS-only response)")

        img_added = 0
        txt_added = 0

        for img in imgs[:max_hits_per_query * 3]:
            src = img.get("src", "")
            if not src:
                srcset = img.get("srcset", "")
                if srcset:
                    src = srcset.split(",")[0].strip().split(" ")[0]

            if "pinimg.com" not in src:
                continue
            if src in seen_imgs:
                continue

            title = (img.get("alt") or "").strip()
            image_results.append({
                "country": country,
                "query": query,
                "source": "pinterest_image",
                "url": src,
                "title": title or f"Pinterest image ({country})",
            })
            seen_imgs.add(src)
            img_added += 1

            if title and len(title) > 25:
                text_results.append({
                    "country": country,
                    "query": query,
                    "source": "pinterest_text",
                    "url": url,
                    "text": title,
                    "score": 1,
                })
                txt_added += 1

            if img_added >= max_hits_per_query:
                break

        # fallback regex extraction (some pages embed URLs in scripts)
        if img_added == 0:
            pinimgs = set(re.findall(r"https://i\.pinimg\.com/[^\"'\s>]+", resp.text))
            for src in list(pinimgs)[:max_hits_per_query]:
                if src in seen_imgs:
                    continue
                image_results.append({
                    "country": country,
                    "query": query,
                    "source": "pinterest_image",
                    "url": src,
                    "title": f"Pinterest image ({country})",
                })
                seen_imgs.add(src)
                img_added += 1

        print(f"    added text={txt_added}, images={img_added}")
        sleep()

    return text_results, image_results


def scrape_mapillary(country: str, bbox: str, token: str, limit: int = 80) -> list[dict]:
    """Fetch geotagged street images from Mapillary Graph API (token required)."""
    results: list[dict] = []
    if not token:
        return results

    endpoint = "https://graph.mapillary.com/images"
    params = {
        "access_token": token,
        "bbox": bbox,
        "limit": min(max(limit, 1), 200),
        "fields": "id,thumb_1024_url,thumb_2048_url,captured_at",
    }

    try:
        resp = requests.get(endpoint, params=params, timeout=25)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception as e:
        print(f"  ! Mapillary [{country}] failed: {e}")
        return results

    print(f"  Mapillary [{country}] raw items: {len(data)}")
    for item in data:
        img_url = item.get("thumb_2048_url") or item.get("thumb_1024_url")
        img_id = item.get("id", "")
        if not img_url and img_id:
            img_url = f"https://graph.mapillary.com/{img_id}/thumb-1024.jpg?access_token={token}"

        if not img_url:
            continue

        results.append({
            "country": country,
            "query": "mapillary_bbox",
            "source": "mapillary",
            "url": img_url,
            "title": f"Mapillary street image {img_id}" if img_id else f"Mapillary street image ({country})",
        })

    print(f"    usable image URLs: {len(results)}")
    return results


# -- YouTube Scraping --------------------------------------------------------

def scrape_youtube(country: str, queries: list[str]) -> tuple[list[dict], list[dict]]:
    """
    Scrape YouTube Data API v3 for video metadata (text) and thumbnails (images).
    Uses google-api-python-client for reliable, quota-aware access.
    
    Returns: (text_results, image_results)
    Quota cost: ~100 units per search query (~100 searches/day on free tier).
    """
    text_results: list[dict] = []
    image_results: list[dict] = []
    seen_video_ids = set()

    if not YOUTUBE_API_KEY:
        print("    ! YouTube: YOUTUBE_API_KEY not set")
        return text_results, image_results

    try:
        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    except ImportError:
        print("    ! YouTube: google-api-python-client not installed. Run: pip install google-api-python-client")
        return text_results, image_results
    except Exception as e:
        print(f"    ! YouTube: failed to build client: {e}")
        return text_results, image_results

    for query in queries:
        print(f"  YouTube: '{query}'")
        try:
            search_response = youtube.search().list(
                q=query,
                type="video",
                part="snippet",
                maxResults=YOUTUBE_MAX_RESULTS,
                relevanceLanguage="en",
                videoEmbeddable="true",
            ).execute()

            items = search_response.get("items", [])
            print(f"    raw results: {len(items)}")

            txt_added = 0
            img_added = 0

            for item in items:
                video_id = item.get("id", {}).get("videoId", "")
                if not video_id or video_id in seen_video_ids:
                    continue
                seen_video_ids.add(video_id)

                snippet = item.get("snippet", {})
                title = snippet.get("title", "")
                description = snippet.get("description", "")
                channel_title = snippet.get("channelTitle", "")
                video_url = f"https://www.youtube.com/watch?v={video_id}"

                # Text: title + description + channel -> sentiment corpus
                combined_text = f"{title}. {description}".strip()
                if combined_text and len(combined_text) > 30:
                    text_results.append({
                        "country": country,
                        "query": query,
                        "source": "youtube_video",
                        "url": video_url,
                        "text": combined_text,
                        "score": 2,  # video descriptions are moderately rich signal
                    })
                    txt_added += 1

                # Image: high-quality thumbnail
                thumbnails = snippet.get("thumbnails", {})
                # Prefer maxres > standard > high > medium
                thumb = (
                    thumbnails.get("maxres") or
                    thumbnails.get("standard") or
                    thumbnails.get("high") or
                    thumbnails.get("medium") or
                    thumbnails.get("default") or
                    {}
                )
                thumb_url = thumb.get("url", "")
                if thumb_url:
                    image_results.append({
                        "country": country,
                        "query": query,
                        "source": "youtube_thumbnail",
                        "url": thumb_url,
                        "title": f"YT: {title[:80]}",
                    })
                    img_added += 1

            print(f"    added text={txt_added}, thumbnails={img_added}")

        except Exception as e:
            print(f"    ! YouTube search error: {e}")

        sleep()

    return text_results, image_results


def scrape_youtube_comments(country: str, queries: list[str], max_comments: int = 50) -> list[dict]:
    """
    Scrape top-level comments from YouTube videos for richer sentiment data.
    Quota cost: ~1 unit per commentThreads.list call.
    Only fetches comments for videos found via scrape_youtube to save quota.
    """
    text_results: list[dict] = []

    if not YOUTUBE_API_KEY:
        return text_results

    try:
        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    except Exception:
        return text_results

    for query in queries:
        print(f"  YouTube comments: '{query}'")
        try:
            # Find videos first
            search_response = youtube.search().list(
                q=query,
                type="video",
                part="snippet",
                maxResults=5,  # Only top 5 videos for comments (save quota)
            ).execute()

            for item in search_response.get("items", []):
                video_id = item.get("id", {}).get("videoId", "")
                if not video_id:
                    continue

                try:
                    comment_response = youtube.commentThreads().list(
                        videoId=video_id,
                        part="snippet",
                        maxResults=max_comments,
                        textFormat="plainText",
                    ).execute()

                    for thread in comment_response.get("items", []):
                        comment = thread.get("snippet", {}).get("topLevelComment", {})
                        comment_text = comment.get("snippet", {}).get("textDisplay", "")
                        comment_url = f"https://www.youtube.com/watch?v={video_id}&lc={comment.get('id', '')}"

                        if comment_text and len(comment_text) > 25:
                            text_results.append({
                                "country": country,
                                "query": query,
                                "source": "youtube_comment",
                                "url": comment_url,
                                "text": comment_text,
                                "score": 1,
                            })

                except Exception as e:
                    # Comments may be disabled for some videos
                    if "commentsDisabled" not in str(e):
                        print(f"    ! Comment fetch error: {e}")
                    continue

        except Exception as e:
            print(f"    ! YouTube comment search error: {e}")

        sleep()

    comment_count = len(text_results)
    if comment_count:
        print(f"    total comments scraped: {comment_count}")
    return text_results


# -- Image Scraping ----------------------------------------------------------

def scrape_ddg_images(country: str, queries: list[str]) -> list[dict]:
    """Fetch image URLs via DDG image search."""
    results = []

    with DDGS() as ddgs:
        for query in queries:
            print(f"  DDG images: '{query}'")
            try:
                hits = list(ddgs.images(query, max_results=MAX_IMG_ITEMS))
            except Exception as e:
                print(f"    ! DDG error: {e}")
                sleep()
                continue

            if not hits:
                print("    ! DDG returned 0 raw hits")
                sleep()
                continue

            sample_keys = sorted(hits[0].keys()) if isinstance(hits[0], dict) else []
            print(f"    raw hits: {len(hits)} | sample keys: {sample_keys}")

            accepted = 0
            missing_url = 0
            for hit in hits:
                if not isinstance(hit, dict):
                    missing_url += 1
                    continue

                # DDG providers can change field names; try common alternatives.
                url = (
                    hit.get("image") or
                    hit.get("url") or
                    hit.get("thumbnail") or
                    hit.get("thumb") or
                    hit.get("src") or
                    hit.get("image_url") or
                    ""
                )

                if url:
                    results.append({
                        "country": country,
                        "query":   query,
                        "source":  "ddg_image",
                        "url":     url,
                        "title":   hit.get("title", ""),
                    })
                    accepted += 1
                else:
                    missing_url += 1

            print(f"    accepted URLs: {accepted} | missing-url hits: {missing_url}")

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

    resolved = 0
    unresolved = 0
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
            resolved += 1
        else:
            unresolved += 1
        time.sleep(0.15)

    print(f"    resolved URLs: {resolved} | unresolved files: {unresolved}")

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
    stats = Counter()
    source_saved = Counter()

    for item in metadata_list:
        stats["total_items"] += 1
        url = item.get("url", "")
        if not url:
            stats["missing_url"] += 1
            continue

        if url in seen_urls:
            stats["duplicate_url"] += 1
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
            stats["already_exists"] += 1
            source_saved[item.get("source", "unknown")] += 1
            continue

        resp = safe_get(url, timeout=30)
        if not resp:
            stats["request_failed"] += 1
            continue

        try:
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            if min(img.size) < MIN_IMG_SIZE:
                stats["too_small"] += 1
                continue

            img.save(filepath, "JPEG", quality=90)
            item["local_path"] = str(filepath)
            item["width"]      = img.size[0]
            item["height"]     = img.size[1]
            successful.append(item)
            stats["saved"] += 1
            source_saved[item.get("source", "unknown")] += 1

        except Exception as e:
            stats["decode_or_save_failed"] += 1
            print(f"    ! Image save failed: {e}")

        time.sleep(0.3)

    print("\nImage download diagnostics:")
    for key in [
        "total_items", "missing_url", "duplicate_url", "already_exists",
        "request_failed", "too_small", "decode_or_save_failed", "saved"
    ]:
        print(f"  - {key}: {stats.get(key, 0)}")
    if source_saved:
        print("  - saved by source:")
        for source, count in source_saved.items():
            print(f"      {source}: {count}")

    return successful


# -- Main --------------------------------------------------------------------

def _run_text_scraping(all_text, all_images, text_queries):
    """Run all text scraping sources for the given queries."""
    print("\n=== TEXT SCRAPING ===")
    for country, queries in text_queries.items():
        print(f"\n[{country.upper()}]")
        items = scrape_ddg_text(country, queries)
        all_text.extend(items)
        print(f"  -> {len(items)} text records")

    # Optional: Mastodon enrichment (text + images)
    if ENABLE_MASTODON:
        print("\n[MASTODON]")
        for country, queries in text_queries.items():
            m_text, m_images = scrape_mastodon(country, queries)
            all_text.extend(m_text)
            all_images.extend(m_images)
            print(f"  [{country}] +{len(m_text)} text, +{len(m_images)} image URLs")
    else:
        print("\n[MASTODON] skipped (ENABLE_MASTODON=false)")

    # Optional: Pinterest enrichment (best-effort)
    if ENABLE_PINTEREST:
        print("\n[PINTEREST]")
        for country, queries in list(text_queries.items()):
            p_text, p_images = scrape_pinterest(country, queries)
            all_text.extend(p_text)
            all_images.extend(p_images)
            print(f"  [{country}] +{len(p_text)} text, +{len(p_images)} image URLs")
    else:
        print("\n[PINTEREST] skipped (ENABLE_PINTEREST=false)")

    # Optional: YouTube enrichment (text + images + comments)
    if ENABLE_YOUTUBE and YOUTUBE_API_KEY:
        print("\n[YOUTUBE]")
        for country, queries in YOUTUBE_QUERIES.items():
            yt_text, yt_images = scrape_youtube(country, queries)
            all_text.extend(yt_text)
            all_images.extend(yt_images)
            print(f"  [{country}] +{len(yt_text)} text, +{len(yt_images)} thumbnails")

        print("\n[YOUTUBE COMMENTS]")
        for country, queries in YOUTUBE_QUERIES.items():
            yt_comments = scrape_youtube_comments(country, queries)
            all_text.extend(yt_comments)
            print(f"  [{country}] +{len(yt_comments)} comments")
    else:
        print("\n[YOUTUBE] skipped")


def _run_image_scraping(all_images, img_queries):
    """Run all image scraping sources for the given queries."""
    print("\n\n=== IMAGE SCRAPING ===")

    # DDG images
    for country, queries in img_queries.items():
        print(f"\n[{country.upper()}] DDG")
        items = scrape_ddg_images(country, queries)
        all_images.extend(items)
        print(f"  -> {len(items)} image URLs")

    # Wikimedia (higher quality, keep separate)
    print("\n[WIKIMEDIA]")
    for country, category in WIKIMEDIA_CATEGORIES.items():
        items = scrape_wikimedia(country, category)
        all_images.extend(items)

    # Optional: Mapillary enrichment (token required)
    if ENABLE_MAPILLARY and MAPILLARY_ACCESS_TOKEN:
        print("\n[MAPILLARY]")
        for country, bbox in MAPILLARY_BBOX.items():
            items = scrape_mapillary(country, bbox, MAPILLARY_ACCESS_TOKEN, limit=MAX_IMG_ITEMS)
            all_images.extend(items)
    else:
        print("\n[MAPILLARY] skipped")


def run():
    print(f"[DIAG] 01_scrape_data DATA_DIR = {DATA_DIR}")
    print(f"[DIAG] 01_scrape_data TEXT_CSV = {TEXT_CSV}")
    print(f"[DIAG] 01_scrape_data IMG_META_CSV = {IMG_META_CSV}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    all_text   = []
    all_images = []

    # ── Round 0: Initial pass with primary queries ──────────────────────────
    print("\n" + "=" * 60)
    print("  ROUND 0: Primary queries")
    print("=" * 60)

    _run_text_scraping(all_text, all_images, TEXT_QUERIES)
    _run_image_scraping(all_images, IMG_QUERIES)

    # ── Retry loop: keep scraping with expanded queries until target reached ─
    seen_text_urls = set()
    seen_img_urls  = set()

    for rnd in range(1, MAX_ROUNDS + 1):
        # Deduplicate & count current totals
        text_df = pd.DataFrame(all_text)
        if not text_df.empty:
            text_df = text_df[text_df["text"].str.strip().str.len() > 30]
            text_df = text_df.drop_duplicates(subset=["url", "source"])

        img_unique_urls = set()
        for img in all_images:
            u = img.get("url", "")
            if u:
                img_unique_urls.add(u)

        total = len(text_df) + len(img_unique_urls)
        print(f"\n{'=' * 60}")
        print(f"  After round {rnd - 1}: text={len(text_df)}, images={len(img_unique_urls)}, total={total}")
        print(f"  Target: {MIN_TOTAL_DATA_POINTS} data points")
        print("=" * 60)

        if total >= MIN_TOTAL_DATA_POINTS:
            print(f"\n  ✓ Target reached ({total} >= {MIN_TOTAL_DATA_POINTS})!")
            break

        print(f"\n  Below target — launching ROUND {rnd} with expanded queries...")

        # Pick expanded queries for this round (cycle through available ones)
        expanded_text = {}
        expanded_img  = {}
        for country in TEXT_QUERIES:
            extra = EXPANDED_TEXT_QUERIES.get(country, [])
            # Pick queries for this round (2 per round, cycling)
            start = (rnd - 1) * 2
            batch = extra[start:start + 2]
            if batch:
                expanded_text[country] = batch

        for country in IMG_QUERIES:
            extra = EXPANDED_IMG_QUERIES.get(country, [])
            start = (rnd - 1) * 2
            batch = extra[start:start + 2]
            if batch:
                expanded_img[country] = batch

        if not expanded_text and not expanded_img:
            print("  No more expanded queries available — stopping retries.")
            break

        _run_text_scraping(all_text, all_images, expanded_text)
        if expanded_img:
            _run_image_scraping(all_images, expanded_img)

    # ── Final save ──────────────────────────────────────────────────────────
    # Save text corpus (deduplicated)
    text_df = pd.DataFrame(all_text)
    if not text_df.empty:
        text_df = text_df[text_df["text"].str.strip().str.len() > 30]
        text_df = text_df.drop_duplicates(subset=["url", "source"])
    text_df.to_csv(TEXT_CSV, index=False)
    print(f"\nText corpus: {len(text_df)} records -> {TEXT_CSV}")
    if not text_df.empty and "country" in text_df.columns:
        print(text_df.groupby(["country", "source"]).size().to_string())

    # Download images
    print(f"\nDownloading {len(all_images)} images...")
    downloaded = download_images(all_images)

    img_df = pd.DataFrame(downloaded)
    img_df.to_csv(IMG_META_CSV, index=False)
    print(f"\nImages: {len(downloaded)} downloaded -> {IMG_META_CSV}")
    print(f"[DIAG] image_metadata exists after write: {IMG_META_CSV.exists()}")
    print(f"[DIAG] image_metadata row count: {len(img_df)}")
    if len(downloaded) == 0:
        print("[WARN] Stage 1 completed with 0 downloaded images.")
        print("[WARN] Dashboard will show no images if image_metadata.csv is empty.")
        print("[WARN] Check DDG raw hits, Wikimedia resolved URLs, and Image download diagnostics above.")
    if not img_df.empty and "country" in img_df.columns:
        print(img_df.groupby(["country", "source"]).size().to_string())

    print("\n=== SCRAPING COMPLETE ===")
    print(f"  Text records : {len(text_df)}")
    print(f"  Images       : {len(downloaded)}")
    print(f"  Total        : {len(text_df) + len(downloaded)}")
    print(f"\nNext: run 02_sentiment_analysis.py")


if __name__ == "__main__":
    run()
