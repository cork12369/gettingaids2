# Manhole Cover Design Analysis Pipeline
# 60.002 Project 2

## Overview

Data-driven design opportunity identification for manhole covers across 7 countries.
The pipeline combines two independent data streams — **NLP sentiment analysis** of
public text (Reddit, travel blogs) and **VLM visual feature extraction** of cover
images — into a correlation model that identifies which design attributes drive
positive public engagement.  All requirements are generated from computed
correlations (no manual guesswork), making every recommendation traceable to a
specific statistical evidence point.

### Research Questions

1. Which countries have the most positively received manhole cover designs?
2. What design vocabulary (aesthetic / functional / cultural / engagement) characterises positive sentiment?
3. How do visual features (ornamentation, cultural motifs, aesthetic appeal, motif diversity) correlate with public engagement?
4. What evidence-based design opportunities and requirements can be extracted from the data?

### Countries Covered

Japan, Singapore, UK, USA, Germany, France, India

---

## How It Works

### Data Flow

```
┌──────────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│ 01_scrape_data   │────▶│ 03_sentiment      │────▶│ 05_cross_analysis    │
│ (Reddit + images)│     │ _analysis.py      │     │                      │
└──────────────────┘     │ (RoBERTa NLP)     │     │ Correlate VLM ×      │
         │               └───────────────────┘     │ sentiment → weights   │
         │                        │                │ + design requirements │
         │                        ▼                └──────────────────────┘
         │               ┌───────────────────┐              │
         └──────────────▶│ 04_image          │──────────────┘
                         │ _processing.py    │
                         │ (VLM + LLM via    │
                         │  OpenRouter)      │
                         └───────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼              ▼
           ┌──────────────┐ ┌──────────┐ ┌──────────────┐
           │ 06_confusion │ │ 07_eval  │ │ 08_generate  │
           │ _matrix.py   │ │ _design  │ │ _report.py   │
           │ (validate    │ │ (score   │ │ (auto-report)│
           │  AI vs human)│ │  new img)│ │              │
           └──────────────┘ └──────────┘ └──────────────┘
```

### VLM + LLM Image Analysis (Stage 3)

Images are analysed by a two-model pipeline via OpenRouter:

1. **VLM** (`rekaai/reka-edge`) — sees the image, extracts visual attributes
   (motifs, ornamentation, style, cultural elements, aesthetic appeal)
2. **LLM** (`bytedance-seed/seed-1.6-flash`) — normalises the VLM output into a
   strict schema, fixing typos and canonicalising values

### Correlation Model (Stage 4)

The cross-analysis produces `design_weights.json` — a weighted scoring function:

```
predicted_sentiment = Σ (vlm_score[attr] × weight[attr])
```

Weights are derived from Pearson correlations between country-aggregated VLM
attributes and public sentiment scores.  For example, if `cultural_elements`
shows r = +0.62 with sentiment, it receives a high weight (~0.35).

### Design Requirements Generation

`generate_design_requirements()` uses three evidence sources to auto-produce
`design_requirements.csv`:

| Source | What it tests |
|--------|---------------|
| Design weights (visual → sentiment) | Which visual attributes boost/hurt sentiment |
| Vocab × visual correlation | How public vocabulary aligns with visual features |
| Hypothesis tests | Industrial→functional, ornate→engagement, aesthetic vocab↔visual |

Every requirement row includes: `id`, `category`, `requirement text`,
`evidence string` (with r-value), `correlation_r`, `direction`, `source`.

---

## Pipeline Stages

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `01_scrape_data.py` | Collect text corpus + images (DuckDuckGo, Wikimedia) | — | `/data/text_raw.csv`, `/data/images/*/` |
| `03_sentiment_analysis.py` | RoBERTa sentiment scoring + keyword analysis | `text_raw.csv` | `text_with_sentiment.csv`, `sentiment_summary.csv`, 5 charts |
| `04_image_processing.py` | VLM visual feature extraction (13 AI attributes) | `/data/images/*/` | `image_analysis.csv`, VLM cache |
| `05_cross_analysis.py` | Correlate visuals × sentiment → model + requirements | `text_with_sentiment.csv` + `image_analysis.csv` | `design_weights.json`, `design_requirements.csv`, `vocab_visual_correlation.json`, 9 figures |
| `06_confusion_matrix.py` | Validate AI sentiment vs human grades | `human_grades.csv` + `grade_sample.csv` | `confusion_matrix.json`, `classification_report.csv` |
| `07_evaluate_design.py` | Score a new design image against the model | `design_weights.json` + new image | Predicted sentiment + country benchmarks |
| `08_generate_report.py` | Auto-generate final report from all outputs | All `/data/output/` files | Markdown/HTML report |

---

## Setup

```bash
pip install -r requirements.txt
```

### API Keys

| Service | Where to get | Free tier | Required for |
|---------|-------------|-----------|--------------|
| Reddit | reddit.com/prefs/apps → "script" | Yes | Stage 1 (text scraping) |
| Flickr | flickr.com/services/apps/create | Yes | Stage 1 (image scraping) |
| OpenRouter | openrouter.ai | Yes (free models) | Stage 3 (VLM image analysis) |

All keys go into Zeabur environment variables — never hardcode them.

---

## Auth

The dashboard is protected by a base32 access key + session cookie.
Rate limiting blocks brute force: 10 attempts/minute, 30/hour per IP.

Generate your keys before deploying:

```bash
# Access key (share with teammates who need dashboard access)
python -c "import base64, os; print(base64.b32encode(os.urandom(10)).decode())"

# Session secret (keep private, never share)
python -c "import secrets; print(secrets.token_hex(32))"
```

Set both in Zeabur environment variables:
```
ACCESS_KEY     = <base32 output>
SESSION_SECRET = <hex output>
```

---

## Directory Structure

```
gettingaids2/
│
├── app.py                    ← Flask dashboard + auth + human grading UI
├── requirements.txt
├── zbpack.json               ← Zeabur build/start config
├── runtime.txt               ← Python 3.12.8
├── .gitignore
├── DOCUMENTATION.md          ← Full technical documentation (843 lines)
├── DEPLOYMENT_PLAN.txt
├── readme.txt                ← This file
│
├── 01_scrape_data.py         ← Stage 1: Reddit + image scraping
├── 03_sentiment_analysis.py  ← Stage 2: RoBERTa sentiment + keyword analysis
├── 04_image_processing.py    ← Stage 3: VLM + LLM visual feature extraction
├── 05_cross_analysis.py      ← Stage 4: Correlation model + design requirements
├── 06_confusion_matrix.py    ← Stage 5: AI vs human validation
├── 07_evaluate_design.py     ← Stage 6: Score new designs against model
├── 08_generate_report.py     ← Stage 7: Auto-generate final report
│
└── data/                     ← Zeabur persistent volume (NOT in repo)
    ├── images/
    │   ├── japan/
    │   ├── singapore/
    │   ├── uk/
    │   ├── usa/
    │   ├── germany/
    │   ├── france/
    │   └── india/
    ├── hf_cache/             ← HuggingFace model cache (survives restarts)
    ├── text_raw.csv
    ├── text_with_sentiment.csv
    ├── image_metadata.csv
    └── output/
        ├── analysis_summary.csv
        ├── cross_analysis.csv
        ├── sentiment_summary.csv
        ├── keyword_frequency.csv
        ├── classification_report.csv
        ├── country_accuracy.csv
        │
        ├── design_weights.json            ← correlation model weights
        ├── vocab_visual_correlation.json  ← text × visual correlation matrix
        ├── design_requirements.csv        ← evidence-backed requirements
        ├── confusion_matrix.json
        │
        ├── sentiment_by_country.png
        ├── sentiment_composition.png
        ├── text_volume_by_country.png
        ├── keyword_heatmap.png
        ├── confidence_distribution.png
        │
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

---

## Run Order (local)

```bash
python 01_scrape_data.py             # ~30 min  → text corpus + images
python 03_sentiment_analysis.py      # ~15 min  → sentiment scores + keyword analysis
python 04_image_processing.py        # ~10 min  → VLM visual features (needs OPENROUTER_API_KEY)
python 05_cross_analysis.py          # ~1  min  → correlation model + design requirements
python 06_confusion_matrix.py        # ~1  min  → confusion matrix (needs human grades)
python 07_evaluate_design.py         # ~2  min  → evaluate new designs against model
python 08_generate_report.py         # ~1  min  → auto-generate report
```

## Run Order (Zeabur)

Open dashboard URL → enter access key → select stage → click Run.
Log streams live and auto-refreshes every 5 seconds.

---

## Key Outputs for Report

### Public Perception
| Output | What it shows |
|--------|--------------|
| `sentiment_heatmap.png` (Fig 6) | Sentiment distribution (neg/neu/pos) by country |
| `sentiment_by_country.png` | Weighted sentiment scores ranked by country |
| `keyword_heatmap.png` | Design vocabulary (aesthetic, functional, cultural…) by country |

### Cross-Analysis (Core Finding)
| Output | What it shows |
|--------|--------------|
| `correlation_heatmap.png` (Fig 8) | Design attribute ↔ sentiment correlation per country |
| `vocab_visual_heatmap.png` (Fig 9) | Text vocabulary × visual attribute correlation with significance |
| `design_weights.json` | Weighted scoring function: `predicted_sentiment = Σ(score×weight)` |
| `design_requirements.csv` | Evidence-backed requirements (DR-01 through DR-NN), each with r-value |

### Validation
| Output | What it shows |
|--------|--------------|
| `confusion_matrix.json` | AI vs human sentiment agreement (Cohen's κ) |
| `human_ai_agreement.png` (Fig 7) | Agreement rate by country |

---

## Novel Contribution

No existing paper has cross-referenced manhole cover visual features with public
sentiment data. All prior CV work on manholes focuses on **defect detection**
(cracks, corrosion), not design quality.

This project establishes a data-driven link between aesthetic design choices and
public engagement through three novel contributions:

1. **Correlation model** — Pearson correlations between VLM-extracted visual
   attributes and NLP sentiment scores, producing an interpretable weighted
   scoring function (no black-box ML).
2. **Evidence-based design requirements** — Automatically generated from
   correlation data with full traceability (every requirement cites its
   statistical evidence: r-value, p-value, source).
3. **Vocabulary–visual cross-correlation** — Links what people *say*
   (aesthetic, functional, cultural vocabulary) with what the VLM *sees*
   (ornamentation, motifs, style), validated through hypothesis testing
   (industrial→functional, ornate→engagement, aesthetic vocab↔aesthetic visual).