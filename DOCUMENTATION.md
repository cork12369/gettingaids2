# Manhole Cover Design Analysis Pipeline — Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Pipeline Stages](#3-pipeline-stages)
4. [Web Dashboard](#4-web-dashboard)
5. [Data Schema](#5-data-schema)
6. [Configuration Reference](#6-configuration-reference)
7. [API Routes](#7-api-routes)
8. [Deployment Guide](#8-deployment-guide)
9. [Development Guide](#9-development-guide)

---

## 1. Project Overview

### Purpose

This project conducts a data-driven design analysis of manhole covers across multiple countries to identify what visual design attributes drive positive public engagement. It combines **NLP sentiment analysis** (analyzing public emotional response from text) with **image processing** (extracting visual features) to find actionable design insights.

### Research Context

**Project:** 60.002 Project 2 (MIT-style engineering course)  
**Repository:** https://github.com/cork12369/gettingaids2.git  
**Deployment:** Zeabur cloud platform

### Countries Covered

| Region | Countries |
|--------|-----------|
| Asia | Japan, Singapore, India, South Korea, Thailand |
| Europe | UK, Germany, France, Italy, Spain, Netherlands |
| Americas | USA, Canada, Brazil, Mexico |
| Oceania | Australia |

### Novel Contribution

This project establishes a **data-driven link between aesthetic design choices and public engagement** — the core design consultancy finding. No existing research has cross-referenced manhole cover visual features with public sentiment data. All prior computer vision work on manholes focuses on defect detection, not design quality.

### Key Research Questions

1. Which countries have the most positively received manhole cover designs?
2. What design vocabulary (aesthetic vs. functional vs. cultural) characterizes positive sentiment?
3. How do visual features correlate with public engagement?
4. What design opportunities exist for improvement?

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GitHub Repository                            │
│    (auto-deploy on push to main branch)                              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Zeabur Cloud Platform                             │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │              Python Service (Flask + Gunicorn)                │   │
│  │                                                               │   │
│  │  ┌─────────────┐  ┌──────────────────────────────────────┐   │   │
│  │  │   app.py    │  │         Pipeline Scripts              │   │   │
│  │  │  Dashboard  │  │  01_scrape_data.py                   │   │   │
│  │  │     +       │  │  03_sentiment_analysis.py            │   │   │
│  │  │    Auth     │  │  04_image_processing.py             │   │   │
│  │  │     +       │  │  05_cross_analysis.py                │   │   │
│  │  │   Grading   │  │  06_confusion_matrix.py              │   │   │
│  │  └─────────────┘  └──────────────────────────────────────┘   │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                │                                    │
│                                ▼                                    │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │            Persistent Volume (/data)                           │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌──────────────────────┐   │   │
│  │  │  /images/   │ │  /output/   │ │   HuggingFace Cache    │   │   │
│  │  │  (500MB+)   │ │  (charts)   │ │   (/hf_cache/)        │   │   │
│  │  └─────────────┘ └─────────────┘ └──────────────────────┘   │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ 01_scrape    │───▶│ 03_sentiment    │───▶│ 05_cross_analysis│
│ _data.py     │    │ _analysis.py    │    │                  │
└──────────────┘    └─────────────────┘    └──────────────────┘
       │                    │                         │
       │                    ▼                         │
       │            ┌──────────────────┐            │
       │            │  Human Grading   │            │
       │            │  (app.py)        │            │
       │            └──────────────────┘            │
       │                    │                         │
       ▼                    ▼                         ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ 04_image    │    │ 06_confusion     │◀───│  Output:         │
│ _processing │    │ _matrix.py       │    │  Visualizations  │
└──────────────┘    └─────────────────┘    │  + CSV Reports   │
       │                    │                └──────────────────┘
       ▼                    ▼
┌──────────────┐    ┌─────────────────┐
│ image_meta  │    │ Human vs AI     │
│ data.csv    │    │ Validation      │
└──────────────┘    └─────────────────┘
```

### Component Responsibilities

| Component | Purpose |
|-----------|---------|
| `app.py` | Web dashboard, authentication, pipeline orchestration, human grading |
| `01_scrape_data.py` | Web scraping (DuckDuckGo, Wikimedia) for text and images |
| `03_sentiment_analysis.py` | NLP sentiment scoring using RoBERTa transformer model |
| `04_image_processing.py` | AI-powered image analysis (VLM + LLM via OpenRouter) |
| `05_cross_analysis.py` | Cross-correlation of sentiment and image data |
| `06_confusion_matrix.py` | Validation of AI sentiment against human grades |

---

## 3. Pipeline Stages

### Stage 1: Data Scraping (`01_scrape_data.py`)

**Purpose:** Collect text corpus and image URLs from multiple keyless sources.

**Data Sources:**
- **Text:** DuckDuckGo search snippets + full-page scraping from travel/blog sites
- **Images:** DuckDuckGo images + Wikimedia Commons categories + fallback sources (Unsplash, Pixabay)

**Key Features:**
- Retry decorator with exponential backoff for reliability
- Rate limiting (1-3 second random delays) for polite scraping
- Country inference via keyword matching (16 countries supported)
- Data balancing to ensure minimum representation per country
- Image validation (minimum 200px dimension) to filter out thumbnails

**Balancing Configuration:**
```python
MIN_TEXT_PER_COUNTRY = 10    # Target minimum text records
MAX_TEXT_PER_COUNTRY = 100   # Cap to prevent over-representation
MIN_IMG_PER_COUNTRY  = 15    # Target minimum images
MAX_IMG_PER_COUNTRY  = 80    # Cap to prevent over-representation
TARGET_IMG_PER_COUNTRY = 50  # Ideal number of images per country
```

**Output Files:**
| File | Description |
|------|-------------|
| `/data/text_raw.csv` | Raw text corpus with country tags |
| `/data/image_metadata.csv` | Image URLs and metadata |
| `/data/images/<country>/` | Downloaded image files |

**Dependencies:** `duckduckgo-search`, `requests`, `beautifulsoup4`, `Pillow`

---

### Stage 2: Sentiment Analysis (`03_sentiment_analysis.py`)

**Purpose:** Analyze sentiment of text corpus using transformer-based NLP.

**Model Used:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Pretrained on Twitter data, optimized for social text
- Outputs: positive (1), neutral (0), negative (-1)
- Confidence scores for each prediction

**Key Features:**
- Country tagging via keyword inference
- Weighted sentiment scoring (upvote-weighted average)
- Design vocabulary analysis (aesthetic, functional, cultural, negative, engagement)
- Batch processing (32 items) to manage memory

**Design Vocabulary Categories:**
| Category | Keywords |
|-----------|----------|
| Aesthetic | beautiful, art, gorgeous, stunning, elegant, intricate |
| Functional | safe, functional, practical, sturdy, dangerous, hazard |
| Cultural | culture, tradition, local, pride, identity, heritage |
| Negative | ugly, boring, dull, plain, eyesore, dirty, rusted |
| Engagement | collect, photograph, hunt, find, discover, tourist |

**Output Files:**
| File | Description |
|------|-------------|
| `/data/text_with_sentiment.csv` | Text with sentiment scores |
| `/data/output/sentiment_summary.csv` | Aggregated sentiment by country |
| `/data/output/keyword_frequency.csv` | Design vocabulary frequency |
| `/data/output/sentiment_by_country.png` | Weighted sentiment chart |
| `/data/output/sentiment_composition.png` | Stacked bar of sentiment types |
| `/data/output/text_volume_by_country.png` | Text samples per country |
| `/data/output/keyword_heatmap.png` | Vocabulary distribution heatmap |
| `/data/output/confidence_distribution.png` | Model confidence boxplot |

**Dependencies:** `transformers`, `torch`, `pandas`, `matplotlib`, `seaborn`

---

### Stage 3: AI-Powered Image Analysis (`04_image_processing.py`)

**Purpose:** Extract visual features from manhole-cover images using a two-model VLM + LLM pipeline via OpenRouter.

**Architecture:**
```
┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐
│  Image file  │───▶│  reka-edge (VLM) │───▶│ seed-1.6-flash(LLM)│
└─────────────┘    │  visual analysis  │    │  schema normalizer  │
                   └──────────────────┘    └────────────────────┘
                            │                       │
                   raw visual JSON          normalized row
                            └───────┬───────────────┘
                                    ▼
                            image_analysis.csv
```

**Models Used:**
| Role | Model | Purpose |
|------|-------|---------|
| VLM | `rekaai/reka-edge` | Visual perception — motif detection, style classification, captioning |
| LLM | `bytedance-seed/seed-1.6-flash` | Schema normalisation — fix typos, canonicalise values, enforce types |

**AI-Extracted Visual Attributes:**
| Attribute | Values | Description |
|-----------|--------|-------------|
| `is_manhole_cover` | bool | Whether image actually shows a manhole cover |
| `relevance_confidence` | 0.0–1.0 | Confidence in relevance classification |
| `image_quality` | low / medium / high | Perceived image quality |
| `view_type` | close-up / medium / street_scene / collage / diagram / other | Camera distance/type |
| `motifs` | pipe-separated list | floral, geometric, animal, mascot, landmark, text, emblem, wave, nature, abstract |
| `ornamentation_level` | plain / minimal / moderate / ornate / highly_ornate | Decorative complexity |
| `symmetry` | none / low / medium / high | Symmetry of design |
| `visual_complexity` | low / medium / high | Overall visual complexity |
| `text_present` | bool | Visible text on cover |
| `cultural_elements` | bool | Culturally specific design features |
| `dominant_style` | traditional / modern / minimalist / artistic / industrial / other | Design style category |
| `aesthetic_appeal` | low / medium / high | Estimated aesthetic quality |
| `caption` | string | One-sentence visual description |

**Key Features:**
- Two-stage pipeline: VLM perception → LLM normalisation
- Resumable processing via MD5-based cache (`image_analysis_cache.json`)
- Automatic retry with exponential backoff (3 attempts per model)
- Rate limiting between batches (configurable batch size)
- Graceful fallback to metadata-only mode when `OPENROUTER_API_KEY` is not set
- Images resized to 1024px max before upload to minimise token cost

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | (none) | OpenRouter API key — enables AI mode when set |
| `VLM_MODEL` | `rekaai/reka-edge` | Vision-language model ID on OpenRouter |
| `LLM_MODEL` | `bytedance-seed/seed-1.6-flash` | Text LLM for normalisation |
| `VLM_MAX_RETRIES` | `3` | Retry attempts per model call |
| `VLM_BATCH_SIZE` | `5` | Images per batch (pause between batches) |
| `VLM_USE_FALLBACK` | `false` | Force metadata-only mode if `"true"` |

**Output Files:**
| File | Description |
|------|-------------|
| `/data/output/image_analysis.csv` | Full image analysis with AI attributes + metadata |
| `/data/output/image_analysis_cache.json` | Resumable VLM+LLM result cache |

**Dependencies:** `Pillow`, `pandas`, `openai` (OpenRouter-compatible client)

---

### Stage 4: Cross Analysis (`05_cross_analysis.py`)

**Purpose:** Correlate sentiment analysis results with image data and generate comprehensive visualizations.

**Key Visualizations:**
| Visualization | Type | Description |
|---------------|------|-------------|
| `text_vs_image_by_country.png` | Grouped bar chart | Text vs image count comparison |
| `sentiment_vs_image_volume.png` | Scatter plot | Identifies engagement opportunities |
| `combined_country_summary.png` | Combo chart | Counts + sentiment overlay |
| `balance_ratio_chart.png` | Bar chart | Text-to-image balance ratio |
| `coverage_summary.png` | Pie charts | Overall coverage distribution |
| `sentiment_heatmap.png` | Heatmap | Sentiment distribution by country |

**Output Files:**
| File | Description |
|------|-------------|
| `/data/output/cross_analysis.csv` | Cross-analysis by country |
| `/data/output/analysis_summary.csv` | Overall statistics |
| `/data/output/cross_analysis_visualizations/` | 6 visualization charts |

**Dependencies:** `pandas`, `matplotlib`, `seaborn`, `numpy`

---

### Stage 5: Confusion Matrix (`06_confusion_matrix.py`)

**Purpose:** Validate AI sentiment predictions against human-graded data.

**Workflow:**
1. Load fixed 2000-snippet sample from `grade_sample.csv`
2. Merge with human grades from `human_grades.csv`
3. Compare human scores (0/1/2) with AI predictions (-1/0/1)
4. Compute confusion matrix and classification metrics

**Metrics Computed:**
| Metric | Description |
|--------|-------------|
| Raw Confusion Matrix | Human rows × AI columns counts |
| Normalized Confusion Matrix | Percentage by true label |
| Overall Accuracy | % of correctly predicted grades |
| Cohen's Kappa (κ) | Agreement measure (0-1 scale) |
| Per-Class Metrics | Precision, Recall, F1 per sentiment class |
| Per-Country Accuracy | Accuracy breakdown by country |

**Cohen's Kappa Interpretation:**
| κ Range | Interpretation |
|---------|----------------|
| 0.8 - 1.0 | Almost perfect agreement |
| 0.6 - 0.8 | Substantial agreement |
| 0.4 - 0.6 | Moderate agreement |
| 0.0 - 0.4 | Fair/slight agreement |

**Output Files:**
| File | Description |
|------|-------------|
| `/data/output/confusion_matrix.png` | Comprehensive visualization figure |
| `/data/output/classification_report.csv` | Precision/Recall/F1 per class |
| `/data/output/country_accuracy.csv` | Per-country accuracy scores |

**Dependencies:** `scikit-learn`, `pandas`, `matplotlib`, `seaborn`

---

## 4. Web Dashboard

### Authentication System

**Login Flow:**
1. User visits dashboard → redirected to `/login`
2. Enter ACCESS_KEY (base32 encoded string)
3. Session cookie set on successful authentication
4. Rate limiting: 10 attempts/minute, 30/hour per IP

**Key Generation:**
```bash
# Access key (for teammates)
python -c "import base64, os; print(base64.b32encode(os.urandom(10)).decode())"

# Session secret (private)
python -c "import secrets; print(secrets.token_hex(32))"
```

### Dashboard Features

**Pipeline Control:**
- Run Full Pipeline (sequential execution of all stages)
- Run individual stages
- Live log streaming with auto-refresh every 5 seconds
- Status indicators for each stage

**Visualization Panels:**
| Tab | Content |
|-----|---------|
| Sentiment | 5 sentiment-related charts |
| Images | 4 image analysis charts |
| Cross Analysis | 6 cross-correlation charts |
| Confusion Matrix | Human vs AI validation |

**Admin Controls (requires ADMIN_RESET_PASSWORD):**
| Action | Description |
|--------|-------------|
| Soft Reset | Clear logs and grading state |
| Reset Analysis | Clear sentiment + image outputs |
| Full Reset | Clear all data including images |
| Reset & Rerun | Full reset then run pipeline |

### Human Grading System

**Purpose:** Create ground truth data to validate AI sentiment predictions.

**Grading Workflow:**
1. Sample generation: 2000-snippet stratified sample from sentiment data
2. Chunk assignment: 10 snippets per chunk, assigned to graders
3. Grading interface: Three buttons (Negative 0, Neutral 1, Positive 2)
4. Keyboard shortcuts: Press 0/1/2 to grade
5. Progress tracking: Team dashboard with per-grader statistics

**Grading Data Files:**
| File | Description |
|------|-------------|
| `grade_sample.csv` | Fixed 2000-snippet sample |
| `grader_assignments.csv` | Chunk assignments by grader |
| `human_grades.csv` | Individual grades submitted |

---

## 5. Data Schema

### Input Data Files

#### `/data/text_raw.csv`
| Column | Type | Description |
|--------|------|-------------|
| country | string | Inferred country/region |
| query | string | Search query used |
| source | string | "ddg_snippet" or "full_page" |
| url | string | Source URL |
| text | string | Text content |
| score | int | Quality weight (1 or 3) |

#### `/data/text_with_sentiment.csv`
| Column | Type | Description |
|--------|------|-------------|
| country | string | Inferred country |
| query | string | Search query used |
| source | string | Source type |
| url | string | Source URL |
| text | string | Text content |
| label | string | "positive", "neutral", "negative" |
| confidence | float | Model confidence (0-1) |
| sentiment | int | Numeric: -1, 0, or 1 |

#### `/data/image_metadata.csv`
| Column | Type | Description |
|--------|------|-------------|
| country | string | Country tag |
| query | string | Search query |
| source | string | "ddg_image", "wikimedia", etc. |
| url | string | Image URL |
| title | string | Image title/description |
| local_path | string | Local file path |
| width | int | Image width (if downloaded) |
| height | int | Image height (if downloaded) |

#### `/data/output/image_analysis.csv`
| Column | Type | Description |
|--------|------|-------------|
| filename | string | File name |
| country | string | Country extracted from path |
| width | int | Image width |
| height | int | Image height |
| format | string | Image format (JPEG, PNG, etc.) |
| mode | string | Color mode (RGB, etc.) |
| aspect_ratio | float | Width/Height ratio |
| file_size_kb | float | File size in KB |
| is_manhole_cover | bool | Whether image shows a manhole cover (AI) |
| relevance_confidence | float | Relevance confidence 0–1 (AI) |
| image_quality | string | low / medium / high (AI) |
| view_type | string | close-up / medium / street_scene / etc. (AI) |
| motifs | string | Pipe-separated motif list (AI) |
| ornamentation_level | string | plain → highly_ornate (AI) |
| symmetry | string | none / low / medium / high (AI) |
| visual_complexity | string | low / medium / high (AI) |
| text_present | bool | Visible text on cover (AI) |
| text_content | string | Transcribed text, if any (AI) |
| cultural_elements | bool | Culturally specific features (AI) |
| cultural_elements_detail | string | Description of cultural elements (AI) |
| dominant_style | string | traditional / modern / artistic / etc. (AI) |
| colour_palette | string | Pipe-separated colour list (AI) |
| aesthetic_appeal | string | low / medium / high (AI) |
| caption | string | One-sentence visual description (AI) |
| vlm_confidence | float | VLM confidence 0–1 (AI) |
| normalization_confidence | float | LLM normalisation confidence 0–1 (AI) |

### Output Data Files

#### `/data/output/sentiment_summary.csv`
| Column | Type | Description |
|--------|------|-------------|
| country | string | Country name |
| weighted_sentiment | float | Score-weighted mean sentiment |
| n_posts | int | Number of posts |
| mean_sentiment | float | Simple mean sentiment |
| pct_positive | float | % positive |
| pct_negative | float | % negative |

#### `/data/output/cross_analysis.csv`
| Column | Type | Description |
|--------|------|-------------|
| country | string | Country name |
| avg_sentiment | float | Mean sentiment score |
| text_count | int | Number of text records |
| avg_confidence | float | Average model confidence |
| image_count | int | Number of images |
| total_image_size_kb | float | Total image size |

#### `/data/output/classification_report.csv`
| Column | Type | Description |
|--------|------|-------------|
| class | string | "Negative", "Neutral", "Positive", "OVERALL" |
| precision | float | Precision score |
| recall | float | Recall score |
| f1_score | float | F1 score |
| support | int | Number of samples |

### Grading Data Files

#### `/data/grade_sample.csv`
| Column | Type | Description |
|--------|------|-------------|
| snippet_id | int | Unique ID (0-1999) |
| country | string | Country tag |
| text | string | Text content |
| sentiment | int | AI sentiment (-1/0/1) |
| label | string | Sentiment label |
| confidence | float | AI confidence |

#### `/data/human_grades.csv`
| Column | Type | Description |
|--------|------|-------------|
| snippet_id | int | Reference to sample |
| grader_id | string | Grader identifier |
| human_score | int | Human grade (0/1/2) |
| timestamp | string | ISO timestamp |

---

## 6. Configuration Reference

### Environment Variables

| Variable | Required | Description |
|---------|----------|-------------|
| `ACCESS_KEY` | Yes | Base32 access key for dashboard login |
| `SESSION_SECRET` | Yes | Flask session signing secret |
| `ADMIN_RESET_PASSWORD` | No | Admin password for reset controls |
| `PORT` | No | Server port (default: 8080) |
| `OPENROUTER_API_KEY` | No | OpenRouter key — enables AI image analysis in Stage 3 |
| `VLM_MODEL` | No | Vision model ID (default: `rekaai/reka-edge`) |
| `LLM_MODEL` | No | Normalisation LLM ID (default: `bytedance-seed/seed-1.6-flash`) |
| `VLM_MAX_RETRIES` | No | Retry attempts per model call (default: 3) |
| `VLM_BATCH_SIZE` | No | Images per batch (default: 5) |
| `VLM_USE_FALLBACK` | No | Set `"true"` to force metadata-only mode |

### Configuration Files

#### `requirements.txt`
```
# Web server
flask>=3.0.0
flask-limiter>=3.5.0
gunicorn>=21.0.0

# Scraping
duckduckgo-search>=6.2.0
requests>=2.31.0
beautifulsoup4>=4.12.0

# Data
pandas>=2.0.0
numpy>=1.24.0

# NLP / Sentiment
transformers>=4.35.0
torch>=2.0.0
sentencepiece>=0.1.99

# Image processing
opencv-python-headless>=4.8.0
scikit-image>=0.21.0
Pillow>=10.0.0
scipy>=1.11.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.13.0

# Metrics
scikit-learn>=1.3.0

# VLM / LLM via OpenRouter
openai>=1.30.0
```

#### `zbpack.json`
```json
{
  "build_command": "sed '/-e/d' requirements.txt | pip install -r /dev/stdin",
  "start_command": "gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 300",
  "output_dir": "/data"
}
```

#### `runtime.txt`
```
python-3.12.8
```

---

## 7. API Routes

### Authentication Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/login` | GET/POST | Login page and authentication |
| `/logout` | GET | Clear session and logout |

### Dashboard Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Main dashboard (requires auth) |
| `/counts` | GET | Get processing counts for status display |
| `/run` | POST | Execute a pipeline stage |
| `/output/<path>` | GET | Serve output files |

### Admin Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/admin/status` | GET | Get pipeline running status |
| `/admin/reset` | POST | Execute reset operation |
| `/admin/logs` | GET | Fetch pipeline logs |

### Grading Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/grade` | GET | Human grading interface |
| `/grade/team` | GET | Team progress dashboard |
| `/grade/api/start` | GET | Start grading session, get chunk |
| `/grade/api/submit` | POST | Submit a grade |
| `/grade/api/progress` | GET | Get team progress statistics |

---

## 8. Deployment Guide

### Prerequisites

- GitHub account with repository
- Zeabur account
- Basic understanding of command line

### Step 1: Prepare Repository

Ensure repository structure:
```
manhole-pipeline/
├── app.py
├── requirements.txt
├── zbpack.json
├── runtime.txt
├── 01_scrape_data.py
├── 03_sentiment_analysis.py
├── 04_image_processing.py
├── 05_cross_analysis.py
└── 06_confusion_matrix.py
```

**Critical:** All scripts use `DATA_DIR = os.getenv('DATA_DIR', '/data')` for paths.
- On Zeabur: set `DATA_DIR=/data` (matches the persistent volume mount)
- Locally: defaults to `./data` if `DATA_DIR` is not set
- `app.py` prints `[DIAG]` lines at startup showing the resolved path and writability

Add to `.gitignore`:
```
data/
output/
*.csv
*.png
hf_cache/
__pycache__/
.env
```

### Step 2: Create Zeabur Project

1. Go to [zeabur.com](https://zeabur.com) → Dashboard → New Project
2. Add Service → Git → Connect GitHub → Select repository
3. Zeabur auto-detects Python from requirements.txt

### Step 3: Mount Persistent Volume

**CRITICAL:** Without this, all scraped data is lost on redeploy.

Zeabur dashboard → Service → Volumes tab:
- Volume ID: `pipeline-data`
- Mount Path: `/data`

### Step 4: Set Environment Variables

| Variable | Required | Value Source |
|----------|----------|-------------|
| `DATA_DIR` | **Yes** | Set to `/data` (must match volume mount path) |
| `ACCESS_KEY` | **Yes** | Generated via `python -c "import base64, os; print(base64.b32encode(os.urandom(10)).decode())"` |
| `SESSION_SECRET` | **Yes** | Generated via `python -c "import secrets; print(secrets.token_hex(32))"` |
| `OPENROUTER_API_KEY` | Recommended | Your OpenRouter API key (enables AI image analysis via VLM+LLM) |

### Step 5: Set Resources

Recommended settings:
- RAM: 2 GB minimum (RoBERTa needs ~1.5GB)
- CPU: 0.5 vCPU (sufficient for CPU inference)

### Step 6: Deploy

Push to GitHub → Zeabur auto-builds and deploys.

**First build:** ~8-12 minutes (torch is ~800MB)  
**Subsequent builds:** ~2-3 minutes (cached layers)

### Running the Pipeline

1. Open Zeabur service URL
2. Enter ACCESS_KEY at login
3. Click "Run Full Pipeline" or individual stages
4. Logs stream live with auto-refresh

---

## 9. Development Guide

### Local Setup

1. **Clone repository:**
```bash
git clone https://github.com/cork12369/gettingaids2.git
cd gettingaids2
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Create local data directories:**
```bash
mkdir -p data/images data/output data/hf_cache
```

5. **Run pipeline stages:**
```bash
python 01_scrape_data.py           # ~20 min
python 03_sentiment_analysis.py   # ~15 min
python 04_image_processing.py     # ~10 min
python 05_cross_analysis.py       # ~1 min
```

6. **Run dashboard locally:**
```bash
export ACCESS_KEY="YOUR_KEY"
export SESSION_SECRET="YOUR_SECRET"
python app.py
```

### Extending the Pipeline

#### Adding New Countries

Edit the `COUNTRY_KEYWORDS` dictionary in both `01_scrape_data.py` and `03_sentiment_analysis.py`:

```python
"new_country": ["newcountry", "nc", "capital_city"],
```

#### Adding New Sentiment Vocabulary

Edit the `DESIGN_VOCAB` dictionary in `03_sentiment_analysis.py`:

```python
"new_category": ["word1", "word2", "word3"],
```

#### Adding New Visualizations

In the appropriate stage file, add a new plotting function:

```python
def plot_new_visualization(data_df, output_dir):
    """Figure X: New Visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    # ... plotting code ...
    plt.savefig(output_dir / "new_visualization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: new_visualization.png")
```

Then call it in the `run()` function.

#### Adding New Pipeline Stages

1. Create new script (e.g., `07_new_stage.py`)
2. Add to dashboard HTML in `app.py`:
   - Add button in `.btn-grid`
   - Add panel in `.graph-panel`
   - Add to `P` object in `loadGraphs()` function
3. Add stage to `stages` array in `runAll()` function

### Troubleshooting

#### Common Issues

| Issue | Solution |
|-------|----------|
| Torch build timeout | Use `torch==2.1.0+cpu` from PyTorch CPU index |
| Rate limiting on scraping | Increase `SCRAPE_DELAY` range |
| Memory errors in sentiment | Reduce batch_size from 32 to 16 |
| Image download failures | Check network connectivity, retry later |
| **Pipeline runs but `/data` is empty** | See "Empty `/data` after pipeline run" below |

#### Empty `/data` after pipeline run

This is the most common deployment issue. Check in this order:

1. **Volume not mounted** — Zeabur dashboard → Service → Volumes tab → ensure Mount Path is `/data`
2. **`DATA_DIR` not set** — Zeabur dashboard → Variables tab → add `DATA_DIR=/data`
3. **Wrong mount path** — the env var `DATA_DIR` must exactly match the volume mount path
4. **Stale build cache** — clear build cache in Zeabur and redeploy
5. **Permission issue** — check startup logs for `[DIAG]` lines:
   ```
   [DIAG] DATA_DIR = /data
   [DIAG] /data writable = True
   ```
   If writable is `False`, the app falls back to `./data` (which is ephemeral and wipes on redeploy).

#### Debug Mode

To see detailed logs:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Appendix: Key Output Files Reference

### Required for Report

| File | Section |
|------|---------|
| `sentiment_by_country.png` | Public Perception Analysis |
| `keyword_heatmap.png` | Qualitative Vocabulary Analysis |
| `image_feature_comparison.png` | Visual Feature Analysis |
| `MAIN_FINDING_sentiment_vs_complexity.png` | Cross-Analysis (novel finding) |
| `design_opportunity_matrix.png` | Design Opportunities |
| Terminal output from script 05 | Design Requirements DR1–DR4 |

### File Locations

All outputs are stored under `DATA_DIR` (default `/data` on Zeabur, `./data` locally):
- Charts: `<DATA_DIR>/output/` (root level)
- Cross Analysis: `<DATA_DIR>/output/cross_analysis_visualizations/`
- Reports: CSV files in `<DATA_DIR>/output/`
- Images: `<DATA_DIR>/images/<country>/`

---

*Documentation Version: 2.1*  
*Last Updated: April 2026*
