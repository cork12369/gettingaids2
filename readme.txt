# Manhole Cover Design Analysis Pipeline
# 60.002 Project 2

## Overview
Data-driven design opportunity identification for manhole covers across 7 countries.
Combines image processing (visual feature extraction) with NLP sentiment analysis
(public emotional response) to find what design attributes drive positive engagement.

## Pipeline Stages

```
01_scrape_reddit.py      → Reddit posts + comments (text corpus)
02_scrape_images.py      → Flickr + Wikimedia images by country
03_sentiment_analysis.py → Sentiment scoring + keyword analysis
04_image_processing.py   → Visual feature extraction
05_cross_analysis.py     → Correlate visuals with sentiment → design requirements
```

## Setup

```bash
pip install -r requirements.txt
```

## API Keys Needed

| Service | Where to get                             | Free tier |
|---------|------------------------------------------|-----------|
| Reddit  | reddit.com/prefs/apps → create "script" | Yes       |
| Flickr  | flickr.com/services/apps/create         | Yes       |

All keys go into Zeabur environment variables — never hardcode them.

## Auth

The dashboard is protected by a base32 access key + session cookie.
Rate limiting blocks brute force: 10 attempts/minute, 30/hour per IP.

Generate your keys before deploying:

  # Access key (share with teammates who need dashboard access)
  python -c "import base64, os; print(base64.b32encode(os.urandom(10)).decode())"

  # Session secret (keep private, never share)
  python -c "import secrets; print(secrets.token_hex(32))"

Set both in Zeabur environment variables:
  ACCESS_KEY     = <base32 output>
  SESSION_SECRET = <hex output>

## Directory Structure

```
manhole_pipeline/
├── app.py                    ← Flask dashboard + auth entrypoint
├── requirements.txt
├── zbpack.json
├── 01_scrape_reddit.py
├── 02_scrape_images.py
├── 03_sentiment_analysis.py
├── 04_image_processing.py
├── 05_cross_analysis.py
└── data/                     ← mounted at /data/ on Zeabur (persistent volume)
    ├── images/
    │   ├── japan/
    │   ├── singapore/
    │   └── ...
    ├── hf_cache/             ← HuggingFace model cache (survives restarts)
    ├── reddit_raw.csv
    ├── reddit_with_sentiment.csv
    ├── image_metadata.csv
    ├── image_features.csv
    └── output/
        ├── sentiment_by_country.png
        ├── keyword_heatmap.png
        ├── image_feature_comparison.png
        ├── MAIN_FINDING_sentiment_vs_complexity.png
        └── design_opportunity_matrix.png
```

## Run Order (local)

```bash
python 01_scrape_reddit.py           # ~20 min
python 02_scrape_images.py           # ~30 min
python 03_sentiment_analysis.py      # ~15 min
python 04_image_processing.py        # ~10 min
python 05_cross_analysis.py          # ~1  min
```

## Run Order (Zeabur)

Open dashboard URL → enter access key → select stage → click Run.
Log streams live and auto-refreshes every 5 seconds.

## Countries Covered
Japan, Singapore, UK, USA, Germany, France, India

## Key Outputs for Report

1. sentiment_by_country.png                  → Section: Public Perception Analysis
2. keyword_heatmap.png                       → Section: Qualitative Vocabulary Analysis
3. image_feature_comparison.png              → Section: Visual Feature Analysis
4. MAIN_FINDING_sentiment_vs_complexity.png  → Section: Cross-Analysis (novel finding)
5. design_opportunity_matrix.png             → Section: Design Opportunities
6. Terminal output from script 05            → Design Requirements DR1–DR4

## Novel Contribution
No existing paper has cross-referenced manhole cover visual features with public
sentiment data. All prior CV work on manholes focuses on defect detection, not
design quality. This analysis establishes a data-driven link between aesthetic
design choices and public engagement — the core design consultancy finding.
