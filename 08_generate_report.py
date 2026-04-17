"""
STAGE 7: Pipeline Report Generator
Consolidates all pipeline outputs into a structured report.

Report sections:
  1. Data Collection      — what we scraped
  2. Sentiment Analysis   — what the public feels
  3. Image Analysis       — what the covers look like
  4. Cross-Analysis       — what design attributes drive sentiment
  5. Design Requirements  — what good covers need to have
  6. Validation           — confusion matrix, human vs AI agreement
  7. Design Evaluation    — applying findings to new designs

Usage:
  python 08_generate_report.py
  python 08_generate_report.py --format pdf
  python 08_generate_report.py --format csv --output /data/output/

Outputs:
  /data/output/pipeline_report.csv
  /data/output/pipeline_report.pdf
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR   = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = DATA_DIR / "output"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("generate_report")

# Source files
TEXT_RAW          = DATA_DIR / "text_raw.csv"
TEXT_SENTIMENT    = DATA_DIR / "text_with_sentiment.csv"
IMAGE_META        = DATA_DIR / "image_metadata.csv"
IMAGE_ANALYSIS    = OUTPUT_DIR / "image_analysis.csv"
CROSS_ANALYSIS    = OUTPUT_DIR / "cross_analysis.csv"
DESIGN_WEIGHTS    = OUTPUT_DIR / "design_weights.json"
CLASSIF_REPORT    = OUTPUT_DIR / "classification_report.csv"
COUNTRY_ACCURACY  = OUTPUT_DIR / "country_accuracy.csv"
HUMAN_GRADES      = DATA_DIR / "human_grades.csv"

# Report outputs
REPORT_CSV = OUTPUT_DIR / "pipeline_report.csv"
REPORT_PDF = OUTPUT_DIR / "pipeline_report.pdf"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_read(path, **kw):
    if path.exists():
        try:
            return pd.read_csv(path, **kw)
        except Exception as e:
            log.warning(f"Could not read {path}: {e}")
    return pd.DataFrame()


def _safe_json(path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def _section_header(title, number):
    return {
        "section": number,
        "subsection": "",
        "metric": f"═══ {title.upper()} ═══",
        "value": "",
        "detail": "",
    }


# ── Section 1: Data Collection ────────────────────────────────────────────────

def section_data_collection():
    rows = [_section_header("1. Data Collection", 1)]

    df_text = _safe_read(TEXT_RAW)
    df_img = _safe_read(IMAGE_META)

    if not df_text.empty:
        rows.append({"section": 1, "subsection": "Text", "metric": "Total text records", "value": len(df_text), "detail": ""})
        countries = df_text["country"].value_counts() if "country" in df_text.columns else pd.Series()
        rows.append({"section": 1, "subsection": "Text", "metric": "Countries covered", "value": len(countries), "detail": ", ".join(countries.index.tolist()[:10])})
        for c, n in countries.items():
            if c != "unknown":
                rows.append({"section": 1, "subsection": "Text by Country", "metric": c, "value": n, "detail": ""})
        if "source" in df_text.columns:
            for s, n in df_text["source"].value_counts().items():
                rows.append({"section": 1, "subsection": "Text Sources", "metric": s, "value": n, "detail": ""})

    if not df_img.empty:
        rows.append({"section": 1, "subsection": "Images", "metric": "Total images downloaded", "value": len(df_img), "detail": ""})
        if "country" in df_img.columns:
            for c, n in df_img["country"].value_counts().items():
                rows.append({"section": 1, "subsection": "Images by Country", "metric": c, "value": n, "detail": ""})

    if df_text.empty and df_img.empty:
        rows.append({"section": 1, "subsection": "", "metric": "Status", "value": "No data yet", "detail": "Run Stage 1 (scrape) first"})

    return rows


# ── Section 2: Sentiment Analysis ─────────────────────────────────────────────

def section_sentiment_analysis():
    rows = [_section_header("2. Sentiment Analysis", 2)]

    df = _safe_read(TEXT_SENTIMENT)

    if df.empty:
        rows.append({"section": 2, "subsection": "", "metric": "Status", "value": "No data", "detail": "Run Stage 2 first"})
        return rows

    # Overall distribution
    if "label" in df.columns:
        counts = df["label"].value_counts()
        total = len(df)
        for lbl, n in counts.items():
            pct = n / total * 100
            rows.append({"section": 2, "subsection": "Overall Sentiment", "metric": str(lbl), "value": n, "detail": f"{pct:.1f}%"})

    # Weighted sentiment by country
    if "country" in df.columns and "sentiment_score" in df.columns:
        country_sent = df.groupby("country").agg(
            avg_sentiment=("sentiment_score", "mean"),
            count=("sentiment_score", "count"),
        ).sort_values("avg_sentiment", ascending=False)

        rows.append({"section": 2, "subsection": "Sentiment by Country", "metric": "Ranked by avg sentiment", "value": len(country_sent), "detail": ""})
        for c, r in country_sent.iterrows():
            if c != "unknown":
                rows.append({"section": 2, "subsection": "Country Sentiment", "metric": c, "value": f"{r['avg_sentiment']:.3f}", "detail": f"n={int(r['count'])}"})

    # Confidence
    if "confidence" in df.columns:
        conf = df["confidence"].dropna()
        if len(conf) > 0:
            rows.append({"section": 2, "subsection": "Confidence", "metric": "Average confidence", "value": f"{conf.mean():.3f}", "detail": f"std={conf.std():.3f}"})

    return rows


# ── Section 3: Image Analysis ─────────────────────────────────────────────────

def section_image_analysis():
    rows = [_section_header("3. Image Analysis", 3)]

    df = _safe_read(IMAGE_ANALYSIS)

    if df.empty:
        rows.append({"section": 3, "subsection": "", "metric": "Status", "value": "No data", "detail": "Run Stage 3 first"})
        return rows

    rows.append({"section": 3, "subsection": "Overview", "metric": "Images analysed", "value": len(df), "detail": ""})

    # Manhole cover detection rate
    if "is_manhole_cover" in df.columns:
        mc = df["is_manhole_cover"].dropna()
        if len(mc) > 0:
            rate = mc.astype(bool).mean() * 100
            rows.append({"section": 3, "subsection": "Detection", "metric": "Manhole cover detection rate", "value": f"{rate:.1f}%", "detail": f"{mc.astype(bool).sum()} of {len(mc)}"})

    # Ornamentation distribution
    if "ornamentation_level" in df.columns:
        orn = df["ornamentation_level"].value_counts()
        for lvl, n in orn.items():
            if pd.notna(lvl):
                rows.append({"section": 3, "subsection": "Ornamentation", "metric": str(lvl), "value": n, "detail": f"{n/len(df)*100:.1f}%"})

    # Aesthetic appeal
    if "aesthetic_appeal" in df.columns:
        aes = df["aesthetic_appeal"].value_counts()
        for lvl, n in aes.items():
            if pd.notna(lvl):
                rows.append({"section": 3, "subsection": "Aesthetic Appeal", "metric": str(lvl), "value": n, "detail": f"{n/len(df)*100:.1f}%"})

    # Cultural elements
    if "cultural_elements" in df.columns:
        ce = df["cultural_elements"].dropna()
        if len(ce) > 0:
            rate = ce.astype(bool).mean() * 100
            rows.append({"section": 3, "subsection": "Cultural Elements", "metric": "Covers with cultural elements", "value": f"{rate:.1f}%", "detail": ""})

    # Dominant style
    if "dominant_style" in df.columns:
        style = df["dominant_style"].value_counts()
        for s, n in style.items():
            if pd.notna(s):
                rows.append({"section": 3, "subsection": "Dominant Style", "metric": str(s), "value": n, "detail": f"{n/len(df)*100:.1f}%"})

    return rows


# ── Section 4: Cross-Analysis ─────────────────────────────────────────────────

def section_cross_analysis():
    rows = [_section_header("4. Cross-Analysis", 4)]

    weights_data = _safe_json(DESIGN_WEIGHTS)

    if not weights_data:
        rows.append({"section": 4, "subsection": "", "metric": "Status", "value": "No data", "detail": "Run Stage 4 (cross-analysis) first"})
        return rows

    weights = weights_data.get("weights", {})
    correlations = weights_data.get("correlations", {})
    benchmarks = weights_data.get("country_benchmarks", {})

    # Design weights
    rows.append({"section": 4, "subsection": "Design Weights", "metric": "Scoring function", "value": "weighted_sum", "detail": "predicted = Σ(score[attr] × weight)"})
    for attr, w in sorted(weights.items(), key=lambda x: -x[1]):
        r = correlations.get(attr, 0)
        rows.append({"section": 4, "subsection": "Weights", "metric": attr, "value": f"{w:.4f}", "detail": f"correlation r={r:+.4f}"})

    # Country benchmarks
    rows.append({"section": 4, "subsection": "Country Benchmarks", "metric": "Countries modelled", "value": len(benchmarks), "detail": ""})
    for country in sorted(benchmarks.keys()):
        bm = benchmarks[country]
        score = sum(bm.get(a, 0) * weights.get(a, 0) for a in weights)
        rows.append({
            "section": 4, "subsection": "Benchmark Scores", "metric": country,
            "value": f"{score:.4f}",
            "detail": f"orn={bm.get('ornamentation_level',0):.2f} cult={bm.get('cultural_elements',0):.2f} aes={bm.get('aesthetic_appeal',0):.2f} motif={bm.get('motif_diversity',0):.2f}"
        })

    return rows


# ── Section 5: Design Requirements ────────────────────────────────────────────

def section_design_requirements():
    rows = [_section_header("5. Design Requirements", 5)]

    weights_data = _safe_json(DESIGN_WEIGHTS)

    if not weights_data:
        rows.append({"section": 5, "subsection": "", "metric": "Status", "value": "No data", "detail": "Run Stage 4 first"})
        return rows

    weights = weights_data.get("weights", {})
    correlations = weights_data.get("correlations", {})

    # Sort by weight (importance)
    sorted_attrs = sorted(weights.items(), key=lambda x: -x[1])

    rows.append({"section": 5, "subsection": "Key Finding", "metric": "Strongest predictor", "value": sorted_attrs[0][0].replace("_", " ").title(), "detail": f"weight={sorted_attrs[0][1]:.2f}, r={correlations.get(sorted_attrs[0][0], 0):+.4f}"})

    recommendations = {
        "ornamentation_level": "Include moderate-to-ornate decorative patterns (borders, radial motifs, geometric bands). Ornamentation is the strongest driver of positive public sentiment.",
        "cultural_elements": "Incorporate region-specific symbols, landmarks, or heritage motifs. Cultural resonance significantly boosts sentiment.",
        "aesthetic_appeal": "Maintain visual balance, color contrast, and symmetry. Aesthetic appeal is a moderate predictor of positive reception.",
        "motif_diversity": "Combine multiple motif types (floral + geometric + text). Diversity adds richness but is the weakest predictor.",
    }

    for i, (attr, w) in enumerate(sorted_attrs, 1):
        r = correlations.get(attr, 0)
        rec = recommendations.get(attr, "Consider including this attribute.")
        rows.append({
            "section": 5, "subsection": f"Requirement #{i}",
            "metric": attr.replace("_", " ").title(),
            "value": f"weight={w:.2f} (r={r:+.4f})",
            "detail": rec,
        })

    # General guidance
    rows.append({"section": 5, "subsection": "Summary", "metric": "Formula", "value": "predicted_sentiment = Σ(score[attr] × weight)", "detail": "Weights derived from correlation with public sentiment data"})
    rows.append({"section": 5, "subsection": "Summary", "metric": "Positive threshold", "value": "≥ 0.55", "detail": "Score above this predicts positive sentiment"})
    rows.append({"section": 5, "subsection": "Summary", "metric": "Negative threshold", "value": "< 0.35", "detail": "Score below this predicts negative sentiment"})

    return rows


# ── Section 6: Validation ─────────────────────────────────────────────────────

def section_validation():
    rows = [_section_header("6. Validation", 6)]

    df_classif = _safe_read(CLASSIF_REPORT)
    df_country = _safe_read(COUNTRY_ACCURACY)
    df_grades = _safe_read(HUMAN_GRADES)

    if df_grades.empty:
        rows.append({"section": 6, "subsection": "", "metric": "Status", "value": "No human grades", "detail": "Complete grading via /grade to generate validation data"})
        return rows

    rows.append({"section": 6, "subsection": "Human Grading", "metric": "Total human grades", "value": len(df_grades), "detail": ""})
    if "grader_id" in df_grades.columns:
        rows.append({"section": 6, "subsection": "Human Grading", "metric": "Unique graders", "value": df_grades["grader_id"].nunique(), "detail": ""})

    if not df_classif.empty:
        rows.append({"section": 6, "subsection": "Classification Report", "metric": "Classes evaluated", "value": len(df_classif), "detail": ""})
        for _, r in df_classif.iterrows():
            cls = r.get("", r.get("class", "?"))
            rows.append({
                "section": 6, "subsection": "Per-Class Metrics",
                "metric": str(cls),
                "value": f"P={r.get('precision',0):.2f} R={r.get('recall',0):.2f} F1={r.get('f1-score',0):.2f}",
                "detail": f"support={r.get('support',0)}",
            })

    if not df_country.empty:
        rows.append({"section": 6, "subsection": "Country Accuracy", "metric": "Countries evaluated", "value": len(df_country), "detail": ""})
        for _, r in df_country.iterrows():
            c = r.get("country", "?")
            acc = r.get("accuracy", 0)
            n = r.get("total", r.get("n", 0))
            rows.append({
                "section": 6, "subsection": "Per-Country Accuracy",
                "metric": c, "value": f"{acc:.2%}", "detail": f"n={n}",
            })

    return rows


# ── Section 7: Design Evaluation Tool ─────────────────────────────────────────

def section_design_evaluation():
    rows = [_section_header("7. Design Evaluation Tool", 7)]

    weights_data = _safe_json(DESIGN_WEIGHTS)

    if not weights_data:
        rows.append({"section": 7, "subsection": "", "metric": "Status", "value": "No model", "detail": "Run full pipeline to build correlation model first"})
        return rows

    weights = weights_data.get("weights", {})
    benchmarks = weights_data.get("country_benchmarks", {})

    rows.append({"section": 7, "subsection": "How It Works", "metric": "Input", "value": "Candidate design image", "detail": "Upload via /evaluate or use 07_evaluate_design.py"})
    rows.append({"section": 7, "subsection": "How It Works", "metric": "Step 1", "value": "VLM analysis", "detail": "Gemini 2.0 Flash extracts design attributes"})
    rows.append({"section": 7, "subsection": "How It Works", "metric": "Step 2", "value": "Attribute encoding", "detail": "Ordinal maps convert categories to 0-1 scores"})
    rows.append({"section": 7, "subsection": "How It Works", "metric": "Step 3", "value": "Weighted scoring", "detail": "predicted_sentiment = Σ(score[attr] × weight)"})
    rows.append({"section": 7, "subsection": "How It Works", "metric": "Step 4", "value": "Benchmark comparison", "detail": f"Compare against {len(benchmarks)} country benchmarks"})
    rows.append({"section": 7, "subsection": "How It Works", "metric": "Step 5", "value": "Recommendation", "detail": "Identify weakest attribute, suggest improvement"})

    # Available benchmarks
    for c in sorted(benchmarks.keys()):
        bm = benchmarks[c]
        score = sum(bm.get(a, 0) * weights.get(a, 0) for a in weights)
        rows.append({"section": 7, "subsection": "Available Benchmarks", "metric": c, "value": f"{score:.4f}", "detail": ""})

    return rows


# ── PDF Generation ────────────────────────────────────────────────────────────

def _render_table(ax, df_section, title=""):
    """Render a DataFrame section as a table on the given axes."""
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", color="#e6edf3",
                     loc="left", pad=12)

    if df_section.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                fontsize=12, color="#8b949e", transform=ax.transAxes)
        return

    cols = [c for c in ["metric", "value", "detail"] if c in df_section.columns]
    show = df_section[cols].copy()
    show.columns = [c.title() for c in cols]

    table = ax.table(
        cellText=show.values,
        colLabels=show.columns,
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(len(cols))))

    # Style
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#30363d")
        if row == 0:
            cell.set_facecolor("#1f6feb")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#161b22" if row % 2 == 0 else "#0d1117")
            cell.set_text_props(color="#c9d1d9")
        cell.set_height(0.06)


def generate_pdf(all_rows):
    """Generate a multi-page PDF report."""
    df = pd.DataFrame(all_rows)

    with PdfPages(str(REPORT_PDF)) as pdf:
        for sec_num in range(1, 8):
            sec_data = df[df["section"] == sec_num]
            if sec_data.empty:
                continue

            # Get section title from first row
            first_metric = sec_data.iloc[0]["metric"] if len(sec_data) > 0 else ""
            section_titles = {
                1: "1. Data Collection",
                2: "2. Sentiment Analysis",
                3: "3. Image Analysis",
                4: "4. Cross-Analysis",
                5: "5. Design Requirements",
                6: "6. Validation",
                7: "7. Design Evaluation Tool",
            }
            title = section_titles.get(sec_num, f"Section {sec_num}")

            # Skip header row for table
            table_data = sec_data[sec_data["subsection"] != ""].copy()
            if table_data.empty and len(sec_data) <= 1:
                # Only header, no data
                fig, ax = plt.subplots(figsize=(11, 7))
                fig.patch.set_facecolor("#0d1117")
                ax.axis("off")
                ax.set_title(title, fontsize=14, fontweight="bold", color="#58a6ff", loc="left")
                ax.text(0.5, 0.4, "No data available — run the relevant pipeline stage first",
                        ha="center", va="center", fontsize=13, color="#8b949e",
                        transform=ax.transAxes)
                pdf.savefig(fig, facecolor=fig.get_facecolor())
                plt.close(fig)
                continue

            fig, ax = plt.subplots(figsize=(11, max(4, len(table_data) * 0.35 + 2)))
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#0d1117")

            _render_table(ax, table_data, title)

            plt.tight_layout()
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)

        # Metadata
        d = pdf.infodict()
        d["Title"] = "Manhole Cover Design Analysis — Pipeline Report"
        d["Author"] = "Automated Pipeline"
        d["Subject"] = f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_report(fmt="both", output_dir=None):
    """Generate the consolidated pipeline report."""
    global OUTPUT_DIR, REPORT_CSV, REPORT_PDF
    if output_dir:
        OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_CSV = OUTPUT_DIR / "pipeline_report.csv"
    REPORT_PDF = OUTPUT_DIR / "pipeline_report.pdf"

    print(f"\n{'='*60}")
    print(f"  Pipeline Report Generator")
    print(f"{'='*60}")

    # Collect all sections
    all_rows = []
    sections = [
        ("Data Collection",     section_data_collection),
        ("Sentiment Analysis",  section_sentiment_analysis),
        ("Image Analysis",      section_image_analysis),
        ("Cross-Analysis",      section_cross_analysis),
        ("Design Requirements", section_design_requirements),
        ("Validation",          section_validation),
        ("Design Evaluation",   section_design_evaluation),
    ]

    for name, func in sections:
        print(f"  Processing: {name}...")
        rows = func()
        all_rows.extend(rows)
        print(f"    → {len(rows)} rows")

    # Save CSV
    if fmt in ("csv", "both"):
        df = pd.DataFrame(all_rows)
        df.to_csv(REPORT_CSV, index=False)
        print(f"\n  [OK] CSV: {REPORT_CSV}")

    # Save PDF
    if fmt in ("pdf", "both"):
        generate_pdf(all_rows)
        print(f"  [OK] PDF: {REPORT_PDF}")

    # Summary
    df = pd.DataFrame(all_rows)
    data_rows = df[df["subsection"] != ""]
    print(f"\n  Report: {len(data_rows)} data rows across {len(sections)} sections")
    print(f"{'='*60}")

    return REPORT_CSV, REPORT_PDF


def main():
    parser = argparse.ArgumentParser(description="Generate consolidated pipeline report")
    parser.add_argument("--format", choices=["csv", "pdf", "both"], default="both",
                        help="Output format (default: both)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: /data/output/)")
    args = parser.parse_args()

    generate_report(fmt=args.format, output_dir=args.output)


if __name__ == "__main__":
    main()