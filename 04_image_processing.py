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
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR       = Path(os.getenv("DATA_DIR", "/data"))
IMAGE_BASE     = DATA_DIR / "images"
OUTPUT_DIR     = DATA_DIR / "output"
OUTPUT_CSV     = OUTPUT_DIR / "image_analysis.csv"
CACHE_FILE     = OUTPUT_DIR / "image_analysis_cache.json"

# Cache versioning — bump if prompts or schema change to invalidate stale entries
CACHE_VERSION  = "v2"

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
You are an expert visual analyst specialising in urban infrastructure photography.
Your task is to analyse images of manhole covers and drainage grates from around the world
and produce structured visual assessments.

Always respond with ONLY valid JSON — no markdown fences, no extra commentary.
"""

VLM_USER_PROMPT = """\
Analyse this image and return a JSON object with EXACTLY these fields:

{
  "is_manhole_cover": true/false,
  "relevance_confidence": 0.0-1.0,
  "image_quality": "low" | "medium" | "high",
  "view_type": "close-up" | "medium" | "street_scene" | "collage" | "diagram" | "other",
  "motifs": ["list of: floral, geometric, animal, mascot, landmark, text, emblem, wave, nature, abstract, none"],
  "ornamentation_level": "plain" | "minimal" | "moderate" | "ornate" | "highly_ornate",
  "symmetry": "none" | "low" | "medium" | "high",
  "visual_complexity": "low" | "medium" | "high",
  "text_present": true/false,
  "text_content": "visible text or empty string",
  "cultural_elements": true/false,
  "cultural_elements_detail": "description or empty string",
  "dominant_style": "traditional" | "modern" | "minimalist" | "artistic" | "industrial" | "other",
  "colour_palette": ["list of dominant colours"],
  "aesthetic_appeal": "low" | "medium" | "high",
  "caption": "one concise descriptive sentence",
  "confidence": 0.0-1.0
}

If the image is NOT a manhole cover, set is_manhole_cover=false and fill what you can.
Respond with ONLY the JSON object."""

LLM_SYSTEM_PROMPT = """\
You are a data-normalisation assistant. Your job is to take raw JSON from a
vision-language model and return a clean, strictly-typed JSON row that matches
the schema exactly. Fix typos, map synonyms to canonical values, and ensure
all required fields are present and correctly typed."""

LLM_USER_PROMPT = """\
Here is raw JSON output from a vision model analysing a manhole-cover image:

---RAW_VLM_JSON---
{raw_json}
---END_RAW_VLM_JSON---

Normalise it into EXACTLY this schema. Only output the JSON object, nothing else:

{{
  "is_manhole_cover": bool,
  "relevance_confidence": float 0.0-1.0,
  "image_quality": "low" | "medium" | "high",
  "view_type": "close-up" | "medium" | "street_scene" | "collage" | "diagram" | "other",
  "motifs": [str],
  "ornamentation_level": "plain" | "minimal" | "moderate" | "ornate" | "highly_ornate",
  "symmetry": "none" | "low" | "medium" | "high",
  "visual_complexity": "low" | "medium" | "high",
  "text_present": bool,
  "text_content": str,
  "cultural_elements": bool,
  "cultural_elements_detail": str,
  "dominant_style": "traditional" | "modern" | "minimalist" | "artistic" | "industrial" | "other",
  "colour_palette": [str],
  "aesthetic_appeal": "low" | "medium" | "high",
  "caption": str,
  "vlm_confidence": float 0.0-1.0,
  "normalization_confidence": float 0.0-1.0
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

    print(f"\n  Output: {OUTPUT_CSV}")
    if ai_mode:
        print(f"  Cache : {CACHE_FILE}")


if __name__ == "__main__":
    process_images()