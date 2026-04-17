"""
STAGE 3: AI-Powered Image Analysis  (VLM + LLM via OpenRouter)

Architecture:
  ┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐
  │  Image file  │───▶│  reka-edge (VLM) │───▶│ seed-1.6-flash(LLM)│
  └─────────────┘    │  visual analysis  │    │  schema normalizer  │
                     └──────────────────┘    └────────────────────┘
                              │                       │
                     raw visual JSON          normalized row
                              │                       │
                              └───────┬───────────────┘
                                      ▼
                              image_analysis.csv

Reads from:
  /data/images/  (per-country folders of scraped manhole-cover images)

Outputs:
  /data/output/image_analysis.csv
  /data/output/image_analysis_cache.json   (resumable VLM+LLM cache)

Environment variables (set in Zeabur / .env):
  OPENROUTER_API_KEY   — required, your OpenRouter key
  VLM_MODEL            — default: rekaai/reka-edge
  LLM_MODEL            — default: bytedance-seed/seed-1.6-flash
  VLM_MAX_RETRIES      — default: 3
  VLM_BATCH_SIZE       — default: 5
  VLM_USE_FALLBACK     — if "true", skip VLM and use metadata-only mode

If OPENROUTER_API_KEY is not set the pipeline still runs in
*metadata-only mode* (legacy behaviour) so existing deployments
do not break.
"""

import os
import re
import json
import time
import base64
import hashlib
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

# Deterministic CV validation
import cv2
from sklearn.cluster import MiniBatchKMeans

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR       = Path(os.getenv("DATA_DIR", "data"))
IMAGE_BASE     = DATA_DIR / "images"
OUTPUT_DIR     = DATA_DIR / "output"
OUTPUT_CSV     = OUTPUT_DIR / "image_analysis.csv"
CACHE_FILE     = OUTPUT_DIR / "image_analysis_cache.json"

# Cache versioning — bump if prompts or schema change to invalidate stale entries
CACHE_VERSION  = "v3"

# OpenRouter
OPENROUTER_KEY  = os.getenv("OPENROUTER_API_KEY", "")
VLM_MODEL       = os.getenv("VLM_MODEL", "rekaai/reka-edge")
LLM_MODEL       = os.getenv("LLM_MODEL", "bytedance-seed/seed-1.6-flash")
OR_BASE_URL     = "https://openrouter.ai/api/v1"
MAX_RETRIES     = int(os.getenv("VLM_MAX_RETRIES", "3"))
BATCH_SIZE      = int(os.getenv("VLM_BATCH_SIZE", "5"))
USE_FALLBACK    = os.getenv("VLM_USE_FALLBACK", "false").lower() == "true"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("image_analysis")

# ── VLM / LLM prompt templates ───────────────────────────────────────────────

VLM_SYSTEM_PROMPT = """\
You are an expert design critic specialising in urban infrastructure and street-level 
visual culture worldwide. You have deep knowledge of:

- Japanese manhole cover art (蓋アート / manhoru-ga), known for colourful, region-specific 
  mascot, landmark, and floral designs that turn utility covers into public art.
- European utility covers, which range from plain industrial plates (UK, Germany) to 
  ornate historical cast-iron patterns (France, Netherlands).
- Covers from the Americas, Southeast Asia, South Korea, and Oceania with varying 
  traditions of functional vs. decorative design.

Your task is to analyse images of manhole covers and drainage grates and produce 
a structured visual assessment as JSON.

RULES:
1. Output ONLY a single JSON object — no markdown fences, no commentary before or after.
2. If the image is clearly NOT a manhole cover or drain grate (e.g., random object, 
   landscape, person), set is_manhole_cover=false and fill fields with null/empty values.
3. For collages or photos containing multiple covers, describe the DOMINANT/central cover.
4. For diagrams or technical drawings, classify view_type as "diagram".
5. Be specific and honest — if a cover is plain, say so; do not inflate scores.
"""

VLM_USER_PROMPT = """\
Examine this image carefully and return a single JSON object with EXACTLY these fields.
Follow the calibration guidelines below each field.

{
  "is_manhole_cover": <boolean>,
  "relevance_confidence": <float 0.0-1.0, how certain you are this is/is not a manhole cover>,
  "image_quality": <"low" | "medium" | "high">,
  "view_type": <"close-up" | "medium" | "street_scene" | "collage" | "diagram" | "other">,
  "motifs": <array of strings from the taxonomy below>,
  "ornamentation_level": <"plain" | "minimal" | "moderate" | "ornate" | "highly_ornate">,
  "symmetry": <"none" | "low" | "medium" | "high">,
  "visual_complexity": <"low" | "medium" | "high">,
  "text_present": <boolean>,
  "text_content": <string: transcribe visible text, or "">,
  "cultural_elements": <boolean>,
  "cultural_elements_detail": <string: describe region-specific or culturally unique features, or "">,
  "dominant_style": <"traditional" | "modern" | "minimalist" | "artistic" | "industrial" | "other">,
  "colour_palette": <array of colour strings, e.g. ["grey","blue","green"]>,
  "aesthetic_appeal": <"low" | "medium" | "high">,
  "caption": <string: one descriptive sentence of 10-25 words>,
  "confidence": <float 0.0-1.0, your overall confidence in this entire assessment>
}

── CALIBRATION GUIDE ──────────────────────────────────────

image_quality:
  "low"    = blurry, pixelated, poor lighting, details hard to discern
  "medium" = acceptable clarity, most features visible but some noise/blur
  "high"   = sharp, well-lit, all design details clearly visible

view_type:
  "close-up"     = cover fills most of the frame, fine details visible
  "medium"       = cover visible with some surrounding context (street, pavement)
  "street_scene" = wide shot where cover is a small part of the scene
  "collage"      = multiple covers shown together in a grid/arrangement
  "diagram"      = technical drawing, blueprint, or schematic
  "other"        = anything that doesn't fit above

motif taxonomy (select ALL that apply from this list):
  "floral"     = flowers, leaves, vines, botanical patterns
  "geometric"  = repeating shapes, grids, concentric circles, abstract geometry
  "animal"     = any animal or creature representation
  "mascot"     = cartoon character, local mascot, or personified figure
  "landmark"   = building, monument, bridge, or recognizable structure
  "text"       = city names, utility labels, brand names, kanji, logos
  "emblem"     = coats of arms, civic crests, corporate logos, official seals
  "wave"       = water patterns, ocean motifs, flowing lines
  "nature"     = landscapes, mountains, rivers, trees, clouds, weather
  "abstract"   = non-representational patterns, color fields, artistic flourishes
  Use ["none"] only if the cover is completely unadorned.

ornamentation_level:
  "plain"         = flat disk or grid, zero intentional decoration
  "minimal"       = one or two simple features (e.g., a single ring, utility text)
  "moderate"      = noticeable pattern or 2-3 decorative elements
  "ornate"        = rich decoration with multiple integrated design elements
  "highly_ornate" = elaborate multi-layer artwork (typical of Japanese manhoru-ga)

symmetry:
  "none"   = no discernible symmetry
  "low"    = roughly balanced but not mirror-identical
  "medium" = bilateral symmetry on at least one axis
  "high"   = near-perfect radial or multi-axis symmetry

visual_complexity (overall information density of the design):
  "low"    = simple shape with minimal visual information
  "medium" = moderate detail; a few distinct visual elements
  "high"   = dense, intricate design with many interlocking elements

dominant_style:
  "traditional" = classic cast-iron patterns, historical motifs, heraldic elements
  "modern"      = clean lines, sans-serif text, contemporary design language
  "minimalist"  = stark simplicity, bare functional surface
  "artistic"    = intentional aesthetic design — murals, illustrations, colour work
  "industrial"  = purely functional — grid/grate pattern, no aesthetic intent
  "other"       = does not fit any category above

aesthetic_appeal (subjective visual attractiveness of the design, not the photo):
  "low"    = unattractive or visually uninteresting design
  "medium" = moderately appealing, some aesthetic merit
  "high"   = visually striking, would draw positive attention from passers-by

If the image is NOT a manhole cover, set is_manhole_cover=false, relevance_confidence 
to your certainty, and set all other fields to null or empty values.
Respond with ONLY the JSON object."""

LLM_SYSTEM_PROMPT = """\
You are a precise data-normalisation engine. You receive raw JSON from a vision-language 
model that analyses manhole-cover images. Your job is to return a clean, strictly-typed 
JSON object matching the canonical schema below.

RULES:
1. Output ONLY the JSON object — no markdown, no commentary.
2. Every field from the schema MUST be present — never drop a field.
3. Map any synonym or non-canonical value to the nearest allowed value.
4. If a field is missing or null in the input, fill it with the default for that type:
   booleans → false, strings → "", floats → 0.0, arrays → [].
5. Strip any field that does NOT appear in the schema.
6. Set normalization_confidence to how confident you are that the output is fully correct:
   1.0 = all values mapped cleanly, 0.5 = some guesses, 0.0 = largely uncertain.

SYNONYM MAPPING (non-exhaustive — apply the same logic to similar cases):
  "very ornate" / "extremely ornate" / "elaborate"    → "highly_ornate"
  "slightly ornate" / "lightly decorated"             → "minimal"
  "decorated" / "medium ornate"                       → "moderate"
  "highly complex" / "very complex"                   → "high"  (visual_complexity)
  "not symmetric" / "asymmetric"                      → "none"  (symmetry)
  "beautiful" / "gorgeous" / "very appealing"         → "high"  (aesthetic_appeal)
  "ugly" / "unappealing"                              → "low"   (aesthetic_appeal)
  "photo" / "photograph" / "snapshot"                 → "close-up" or "medium" (view_type)
  "Japanese style" / "anime" / "colorful"             → "artistic" (dominant_style)
  "Cast iron" / "cast-iron" / "iron"                  → "industrial" (dominant_style)
  Boolean synonyms: "yes"/"true"/"1" → true, "no"/"false"/"0" → false
"""

LLM_USER_PROMPT = """\
Here is raw JSON from the vision model:

---RAW VLM OUTPUT---
{raw_json}
---END RAW VLM OUTPUT---

Normalise it into EXACTLY this schema (output only the JSON):

{{
  "is_manhole_cover": <bool>,
  "relevance_confidence": <float 0.0-1.0>,
  "image_quality": "low" | "medium" | "high",
  "view_type": "close-up" | "medium" | "street_scene" | "collage" | "diagram" | "other",
  "motifs": [<str from: floral, geometric, animal, mascot, landmark, text, emblem, wave, nature, abstract>],
  "ornamentation_level": "plain" | "minimal" | "moderate" | "ornate" | "highly_ornate",
  "symmetry": "none" | "low" | "medium" | "high",
  "visual_complexity": "low" | "medium" | "high",
  "text_present": <bool>,
  "text_content": <str>,
  "cultural_elements": <bool>,
  "cultural_elements_detail": <str>,
  "dominant_style": "traditional" | "modern" | "minimalist" | "artistic" | "industrial" | "other",
  "colour_palette": [<str>],
  "aesthetic_appeal": "low" | "medium" | "high",
  "caption": <str>,
  "vlm_confidence": <float 0.0-1.0, copy from input "confidence" field>,
  "normalization_confidence": <float 0.0-1.0, your confidence in this normalization>
}}"""


# ── OpenRouter helpers ────────────────────────────────────────────────────────

def _get_client():
    """Lazy-init an OpenAI client pointed at OpenRouter."""
    from openai import OpenAI
    return OpenAI(base_url=OR_BASE_URL, api_key=OPENROUTER_KEY)


def _encode_image(image_path: Path) -> str:
    """Return base64-encoded JPEG of a resized image."""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((1024, 1024), Image.LANCZOS)
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_json(text: str) -> dict:
    """Extract the first valid JSON object from model output.
    Handles markdown fences, leading/trailing prose, and nested braces."""
    text = text.strip()
    # Strip markdown fences
    if "```" in text:
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first { ... } using brace matching
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                return json.loads(candidate)
    raise json.JSONDecodeError("Unbalanced braces in JSON", text, start)


# ── Schema validation ─────────────────────────────────────────────────────────

VALID_ENUMS = {
    "image_quality":         {"low", "medium", "high"},
    "view_type":             {"close-up", "medium", "street_scene", "collage", "diagram", "other"},
    "ornamentation_level":   {"plain", "minimal", "moderate", "ornate", "highly_ornate"},
    "symmetry":              {"none", "low", "medium", "high"},
    "visual_complexity":     {"low", "medium", "high"},
    "dominant_style":        {"traditional", "modern", "minimalist", "artistic", "industrial", "other"},
    "aesthetic_appeal":      {"low", "medium", "high"},
}

def _coerce_list(value) -> list:
    """Ensure a value is a list. Handles strings, None, and already-list values."""
    if value is None:
        return []
    if isinstance(value, str):
        if "," in value or "|" in value:
            return [v.strip() for v in value.replace("|", ",").split(",") if v.strip()]
        return [value] if value else []
    if isinstance(value, list):
        return value
    return [str(value)]


def _validate_row(data: dict) -> dict:
    """Canonicalise enum fields, coerce list fields, clamp floats."""
    out = dict(data)
    for field, allowed in VALID_ENUMS.items():
        val = out.get(field, "")
        if isinstance(val, str):
            val_lower = val.lower().strip().replace(" ", "_")
            if val_lower not in allowed:
                matches = [a for a in allowed if a in val_lower or val_lower in a]
                val_lower = matches[0] if matches else "other"
            out[field] = val_lower
        else:
            out[field] = "other"
    for field in ("motifs", "colour_palette"):
        out[field] = _coerce_list(out.get(field))
    for field in ("is_manhole_cover", "text_present", "cultural_elements"):
        val = out.get(field)
        if isinstance(val, str):
            out[field] = val.lower() in ("true", "yes", "1")
        elif val is None:
            out[field] = None
        else:
            out[field] = bool(val)
    for field in ("relevance_confidence", "vlm_confidence", "normalization_confidence", "confidence"):
        val = out.get(field)
        if val is not None:
            try:
                out[field] = max(0.0, min(1.0, float(val)))
            except (ValueError, TypeError):
                out[field] = None
    for field in ("text_content", "cultural_elements_detail", "caption"):
        val = out.get(field)
        out[field] = str(val) if val else ""
    return out


def _call_vlm(client, image_path: Path) -> dict:
    """Call the VLM model with an image and return parsed JSON."""
    b64 = _encode_image(image_path)
    ext = image_path.suffix.lstrip(".").upper()
    mime = f"image/{'JPEG' if ext in ('JPG','JPEG') else ext}"

    resp = client.chat.completions.create(
        model=VLM_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": VLM_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": VLM_USER_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{b64}"
                }}
            ]}
        ]
    )
    text = resp.choices[0].message.content.strip()
    return _extract_json(text)


def _call_llm(client, raw_vlm_json: str) -> dict:
    """Call the LLM normaliser and return parsed JSON."""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": LLM_USER_PROMPT.format(raw_json=raw_vlm_json)}
        ]
    )
    text = resp.choices[0].message.content.strip()
    return _extract_json(text)


def analyze_image_ai(client, image_path: Path, cache: dict) -> dict:
    """Run VLM → LLM pipeline for a single image, with caching and retry."""
    raw_bytes = image_path.read_bytes()
    img_hash = hashlib.md5(
        f"{CACHE_VERSION}:{VLM_MODEL}:{LLM_MODEL}:".encode() + raw_bytes
    ).hexdigest()

    if img_hash in cache:
        log.info(f"  cache hit: {image_path.name}")
        return cache[img_hash]

    # ── VLM call with retry ───────────────────────────────────────────────
    vlm_result = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            vlm_result = _call_vlm(client, image_path)
            break
        except json.JSONDecodeError as e:
            log.warning(f"  VLM JSON parse error (attempt {attempt}): {e}")
        except Exception as e:
            log.warning(f"  VLM error (attempt {attempt}): {e}")
            time.sleep(2 ** attempt)
    if vlm_result is None:
        vlm_result = {"error": "VLM_FAILED", "is_manhole_cover": None, "confidence": 0}

    # ── LLM normalisation with retry ──────────────────────────────────────
    llm_result = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            llm_result = _call_llm(client, json.dumps(vlm_result))
            break
        except json.JSONDecodeError as e:
            log.warning(f"  LLM JSON parse error (attempt {attempt}): {e}")
        except Exception as e:
            log.warning(f"  LLM error (attempt {attempt}): {e}")
            time.sleep(2 ** attempt)
    if llm_result is None:
        llm_result = vlm_result  # fall back to raw VLM output

    # Merge: start with VLM, overlay LLM normalised fields
    merged = {**vlm_result, **llm_result}

    # Validate and canonicalise the merged result
    merged = _validate_row(merged)

    cache[img_hash] = merged

    # Persist cache incrementally
    CACHE_FILE.write_text(json.dumps(cache, indent=2))
    return merged


# ── Metadata-only helpers (legacy fallback) ───────────────────────────────────

def extract_metadata(image_path: Path) -> dict:
    """Fast local-only metadata extraction (no AI calls)."""
    stat = image_path.stat()
    with Image.open(image_path) as img:
        w, h = img.size
        mode = img.mode
        fmt = (img.format or "").upper()
    return {
        "width": w,
        "height": h,
        "format": fmt,
        "mode": mode,
        "aspect_ratio": round(w / max(h, 1), 3),
        "file_size_kb": round(stat.st_size / 1024, 2),
    }


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


# ── Deterministic CV Validation ───────────────────────────────────────────────

# HSV hue bins for colour naming
_HSV_COLOUR_MAP = [
    (10,  "red"),     # H 0–10
    (25,  "orange"),  # H 10–25
    (35,  "yellow"),  # H 25–35
    (85,  "green"),   # H 35–85
    (130, "blue"),    # H 85–130
    (170, "purple"),  # H 130–170
    (180, "red"),     # H 170–180 wraps to red
]

def _hsv_to_colour_name(h: float, s: float, v: float) -> str:
    """Map an HSV pixel to a named colour."""
    if v < 20:
        return "black"
    if s < 30:
        return "grey" if v < 220 else "white"
    for upper, name in _HSV_COLOUR_MAP:
        if h < upper:
            return name
    return "grey"


def cv_edge_density(image_path: Path) -> dict:
    """Compute edge density via Canny detection for visual complexity validation.

    Returns:
        dict with keys: cv_edge_density (float 0-1), cv_complexity_score (float),
                        cv_complexity_label (str: low/medium/high)
    """
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"cv_edge_density": None, "cv_complexity_score": None,
                    "cv_complexity_label": None}

        # Resize to standardised dimensions for consistency
        h, w = img.shape
        scale = min(1.0, 800 / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

        # Denoise then detect edges
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Edge density = ratio of edge pixels to total
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_pixels = int(np.count_nonzero(edges))
        density = edge_pixels / total_pixels

        # Map density → label
        if density < 0.05:
            label = "low"
        elif density < 0.15:
            label = "medium"
        else:
            label = "high"

        return {
            "cv_edge_density": round(density, 4),
            "cv_complexity_score": round(density, 4),
            "cv_complexity_label": label,
        }
    except Exception as e:
        log.warning(f"  CV edge analysis failed: {e}")
        return {"cv_edge_density": None, "cv_complexity_score": None,
                "cv_complexity_label": None}


def cv_color_analysis(image_path: Path) -> dict:
    """Extract dominant HSV colour clusters via KMeans to validate VLM colour palette.

    Returns:
        dict with keys: cv_color_palette (str, pipe-separated), cv_color_count (int),
                        cv_dominant_colors (list of str)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return {"cv_color_palette": "", "cv_color_count": 0, "cv_dominant_colors": []}

        # Resize for speed (max 300px on longest side)
        h, w = img.shape[:2]
        scale = min(1.0, 300 / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

        # Convert BGR → HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Reshape to pixel list
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        # Filter out near-black (V < 20) and near-white (V > 235, S < 20)
        mask = ~((pixels[:, 2] < 20) | ((pixels[:, 2] > 235) & (pixels[:, 1] < 20)))
        filtered = pixels[mask]

        if len(filtered) < 10:
            # Too few coloured pixels — likely greyscale image
            grey_pixels = pixels[~mask]
            if len(grey_pixels) > 0:
                avg_v = float(grey_pixels[:, 2].mean())
                if avg_v < 80:
                    return {"cv_color_palette": "black", "cv_color_count": 1,
                            "cv_dominant_colors": ["black"]}
                elif avg_v < 180:
                    return {"cv_color_palette": "grey", "cv_color_count": 1,
                            "cv_dominant_colors": ["grey"]}
                else:
                    return {"cv_color_palette": "white", "cv_color_count": 1,
                            "cv_dominant_colors": ["white"]}
            return {"cv_color_palette": "", "cv_color_count": 0, "cv_dominant_colors": []}

        # Subsample if too many pixels (speed)
        if len(filtered) > 10000:
            indices = np.random.choice(len(filtered), 10000, replace=False)
            filtered = filtered[indices]

        # KMeans clustering (k=5)
        n_clusters = min(5, len(filtered))
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                                  batch_size=1000, n_init=3)
        labels = kmeans.fit_predict(filtered)
        cluster_sizes = np.bincount(labels)

        # Map cluster centroids to colour names
        colour_counts = {}
        for idx in range(n_clusters):
            centroid = kmeans.cluster_centers_[idx]
            h_val, s_val, v_val = centroid[0], centroid[1], centroid[2]
            name = _hsv_to_colour_name(h_val, s_val, v_val)
            if name not in colour_counts:
                colour_counts[name] = 0
            colour_counts[name] += int(cluster_sizes[idx])

        # Sort by prevalence (most dominant first)
        sorted_colours = sorted(colour_counts.items(), key=lambda x: -x[1])
        dominant = [c for c, _ in sorted_colours]

        return {
            "cv_color_palette": "|".join(dominant),
            "cv_color_count": len(dominant),
            "cv_dominant_colors": dominant,
        }
    except Exception as e:
        log.warning(f"  CV colour analysis failed: {e}")
        return {"cv_color_palette": "", "cv_color_count": 0, "cv_dominant_colors": []}


def _jaccard_similarity(set_a, set_b) -> float:
    """Compute Jaccard similarity between two sets of colour names."""
    a = set(c.lower().strip() for c in set_a if c)
    b = set(c.lower().strip() for c in set_b if c)
    if not a and not b:
        return 1.0  # Both empty — perfect match
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_images():
    """Analyse all downloaded images — AI-powered when API key available."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover images
    if not IMAGE_BASE.exists():
        print(f"WARNING: {IMAGE_BASE} not found. Run 01_scrape_data.py first.")
        return

    image_files = sorted(
        p for p in IMAGE_BASE.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
        and is_valid_image(p)
    )

    if not image_files:
        print("WARNING: No valid images found.")
        return

    print(f"Found {len(image_files)} valid images across countries.\n")

    # Determine mode
    ai_mode = bool(OPENROUTER_KEY) and not USE_FALLBACK
    client = None
    cache: dict = {}

    if ai_mode:
        try:
            client = _get_client()
            if CACHE_FILE.exists():
                cache = json.loads(CACHE_FILE.read_text())
                print(f"Loaded cache with {len(cache)} entries.")
            print(f"AI mode: VLM={VLM_MODEL}  LLM={LLM_MODEL}")
        except Exception as e:
            log.error(f"Failed to init OpenRouter client: {e}")
            ai_mode = False
            print("Falling back to metadata-only mode.\n")

    # Process each image
    rows = []
    for i, img_path in enumerate(image_files, 1):
        country = img_path.parent.name
        print(f"[{i}/{len(image_files)}] {country}/{img_path.name}")

        meta = extract_metadata(img_path)
        row = {
            "filename": img_path.name,
            "country": country,
            **meta,
        }

        if ai_mode and client:
            try:
                ai = analyze_image_ai(client, img_path, cache)
                row.update({
                    "is_manhole_cover":       ai.get("is_manhole_cover"),
                    "relevance_confidence":   ai.get("relevance_confidence"),
                    "image_quality":          ai.get("image_quality"),
                    "view_type":              ai.get("view_type"),
                    "motifs":                 "|".join(ai.get("motifs", [])),
                    "ornamentation_level":    ai.get("ornamentation_level"),
                    "symmetry":               ai.get("symmetry"),
                    "visual_complexity":      ai.get("visual_complexity"),
                    "text_present":           ai.get("text_present"),
                    "text_content":           ai.get("text_content", ""),
                    "cultural_elements":      ai.get("cultural_elements"),
                    "cultural_elements_detail": ai.get("cultural_elements_detail", ""),
                    "dominant_style":         ai.get("dominant_style"),
                    "colour_palette":         "|".join(ai.get("colour_palette", [])),
                    "aesthetic_appeal":       ai.get("aesthetic_appeal"),
                    "caption":                ai.get("caption", ""),
                    "vlm_confidence":         ai.get("vlm_confidence", ai.get("confidence")),
                    "normalization_confidence": ai.get("normalization_confidence"),
                    "ai_pipeline":            "ok",
                })
                print(f"    ✓ AI: manhole={ai.get('is_manhole_cover')}  "
                      f"ornamentation={ai.get('ornamentation_level')}  "
                      f"style={ai.get('dominant_style')}")
            except Exception as e:
                log.error(f"    AI analysis failed: {e}")
                row["ai_error"] = str(e)
                row["ai_pipeline"] = "error"

        # ── Deterministic CV validation (always runs) ──────────────────────
        cv_edge = cv_edge_density(img_path)
        cv_colour = cv_color_analysis(img_path)
        row.update(cv_edge)
        row.update({
            "cv_color_palette":  cv_colour["cv_color_palette"],
            "cv_color_count":    cv_colour["cv_color_count"],
        })

        # Validate VLM vs CV
        vlm_complexity = row.get("visual_complexity", "")
        cv_complexity = cv_edge.get("cv_complexity_label")
        row["complexity_agreement"] = (
            bool(vlm_complexity) and bool(cv_complexity)
            and vlm_complexity.lower() == cv_complexity.lower()
        ) if vlm_complexity and cv_complexity else None

        vlm_palette = row.get("colour_palette", "")
        if isinstance(vlm_palette, str) and vlm_palette:
            vlm_colours = set(c.strip().lower() for c in vlm_palette.split("|") if c.strip())
        else:
            vlm_colours = set()
        cv_colours = set(c.lower() for c in cv_colour.get("cv_dominant_colors", []))
        row["color_palette_overlap"] = round(_jaccard_similarity(vlm_colours, cv_colours), 3)

        cv_log_parts = []
        if cv_edge.get("cv_edge_density") is not None:
            cv_log_parts.append(f"edge_density={cv_edge['cv_edge_density']}")
            cv_log_parts.append(f"complexity={cv_edge['cv_complexity_label']}")
        if cv_colour.get("cv_color_count"):
            cv_log_parts.append(f"colors={cv_colour['cv_color_count']}")
        if row["color_palette_overlap"] is not None:
            cv_log_parts.append(f"overlap={row['color_palette_overlap']}")
        if cv_log_parts:
            print(f"    ✓ CV: " + "  ".join(cv_log_parts))

        rows.append(row)

        # Rate-limit pause between AI calls
        if ai_mode and i % BATCH_SIZE == 0:
            time.sleep(1)

    # Build DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Stage 3 Complete  ({'AI-powered' if ai_mode else 'metadata-only'})")
    print(f"{'='*55}")
    print(f"  Images analysed : {len(df)}")
    print(f"  Countries       : {df['country'].nunique()}")
    print(f"  Avg file size   : {df['file_size_kb'].mean():.1f} KB")
    print(f"  Avg dimensions  : {df['width'].mean():.0f} × {df['height'].mean():.0f}")

    if ai_mode and "is_manhole_cover" in df.columns:
        valid = df["is_manhole_cover"].notna()
        if valid.any():
            mc_rate = df.loc[valid, "is_manhole_cover"].astype(bool).mean() * 100
            print(f"  Manhole covers  : {mc_rate:.1f}% of analysed images")

    # CV validation summary
    if "cv_edge_density" in df.columns:
        edge_valid = df["cv_edge_density"].notna()
        if edge_valid.any():
            avg_density = df.loc[edge_valid, "cv_edge_density"].mean()
            print(f"  CV Edge Density : avg={avg_density:.4f}")
    if "complexity_agreement" in df.columns:
        agree_valid = df["complexity_agreement"].notna()
        if agree_valid.any():
            agree_rate = df.loc[agree_valid, "complexity_agreement"].mean() * 100
            print(f"  VLM-CV Agreement: {agree_rate:.1f}% (visual_complexity)")
    if "color_palette_overlap" in df.columns:
        overlap_valid = df["color_palette_overlap"].notna()
        if overlap_valid.any():
            avg_overlap = df.loc[overlap_valid, "color_palette_overlap"].mean()
            print(f"  Color Overlap   : avg Jaccard={avg_overlap:.3f}")

    print(f"\n  Output: {OUTPUT_CSV}")
    if ai_mode:
        print(f"  Cache : {CACHE_FILE}")


if __name__ == "__main__":
    process_images()