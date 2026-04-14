# Manhole Cover Design Analysis Pipeline
# 60.002 Project 2

## Overview
Data-driven design opportunity identification for manhole covers across 7 countries.
Combines image processing (visual feature extraction) with NLP sentiment analysis
(public emotional response) to find what design attributes drive positive engagement.

## Pipeline Stages

```
01_scrape_data.py        → Reddit + image scraping (text corpus + images)
03_sentiment_analysis.py → Sentiment scoring + keyword analysis
04_image_processing.py   → Visual feature extraction (VLM)
05_cross_analysis.py     → Correlate visuals with sentiment → design requirements
06_confusion_matrix.py   → Sentiment model confusion matrix + metrics
07_evaluate_design.py    → Score new designs against correlation model
08_generate_report.py    → Auto-generate final report from all outputs
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
├── runtime.txt               ← Python version for Zeabur
├── .gitignore
├── DOCUMENTATION.md          ← Full technical documentation
├── DEPLOYMENT_PLAN.txt
├── 01_scrape_data.py         ← Reddit + image scraping
├── 03_sentiment_analysis.py  ← Sentiment scoring + keyword analysis
├── 04_image_processing.py    ← VLM visual feature extraction
├── 05_cross_analysis.py      ← Correlate visuals × sentiment → design requirements
├── 06_confusion_matrix.py    ← Sentiment model confusion matrix + metrics
├── 07_evaluate_design.py     ← Score new designs against correlation model
├── 08_generate_report.py     ← Auto-generate final report
└── data/                     ← mounted at /data/ on Zeabur (persistent volume)
    ├── images/
    │   ├── japan/
    │   ├── singapore/
    │   └── ...
    ├── hf_cache/             ← HuggingFace model cache (survives restarts)
    ├── text_with_sentiment.csv
    ├── image_analysis.csv
    └── output/
        ├── analysis_summary.csv
        ├── cross_analysis.csv
        ├── sentiment_summary.csv
        ├── design_weights.json            ← correlation model (weighted scoring)
        ├── vocab_visual_correlation.json  ← text × visual correlation matrix
        ├── design_requirements.csv        ← evidence-backed design requirements
        ├── confusion_matrix.json
        └── cross_analysis_visualizations/
            ├── text_vs_image_by_country.png      (Figure 1)
            ├── sentiment_vs_image_volume.png     (Figure 2)
            ├── combined_country_summary.png       (Figure 3)
            ├── balance_ratio_chart.png            (Figure 4)
            ├── coverage_summary.png               (Figure 5)
            ├── sentiment_heatmap.png              (Figure 6)
            ├── human_ai_agreement.png             (Figure 7)
            ├── correlation_heatmap.png            (Figure 8)
            └── vocab_visual_heatmap.png           (Figure 9)
```

## Run Order (local)

```bash
python 01_scrape_data.py             # ~30 min  (Reddit + images)
python 03_sentiment_analysis.py      # ~15 min
python 04_image_processing.py        # ~10 min
python 05_cross_analysis.py          # ~1  min  (generates design_weights.json + design_requirements.csv)
python 06_confusion_matrix.py        # ~1  min
python 07_evaluate_design.py         # ~2  min  (evaluate new designs against model)
python 08_generate_report.py         # ~1  min
```

## Run Order (Zeabur)

Open dashboard URL → enter access key → select stage → click Run.
Log streams live and auto-refreshes every 5 seconds.

## Countries Covered
Japan, Singapore, UK, USA, Germany, France, India

## Key Outputs for Report

1. sentiment_heatmap.png (Fig 6)            → Section: Public Perception by Country
2. correlation_heatmap.png (Fig 8)          → Section: Design–Sentiment Correlation Model
3. vocab_visual_heatmap.png (Fig 9)         → Section: Text × Visual Cross-Correlation
4. design_weights.json                      → Section: Weighted Scoring Function
5. design_requirements.csv                  → Section: Evidence-Based Design Requirements
6. vocab_visual_correlation.json            → Section: Hypothesis Testing Results
7. confusion_matrix.json                    → Section: Sentiment Model Validation
8. 08_generate_report.py output             → Section: Final Auto-Generated Report

## Novel Contribution
No existing paper has cross-referenced manhole cover visual features with public
sentiment data. All prior CV work on manholes focuses on defect detection, not
design quality. This analysis establishes a data-driven link between aesthetic
design choices and public engagement — the core design consultancy finding.
