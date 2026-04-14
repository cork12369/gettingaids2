"""
STAGE 5: Design Evaluation
Evaluate candidate design images against the design–sentiment correlation model.

Reuses Stage 3 VLM+LLM infrastructure (04_image_processing.py) for visual analysis
and Stage 4 correlation model (05_cross_analysis.py → design_weights.json) for scoring.

Usage:
  python 07_evaluate_design.py --images design_a.png design_b.jpg
  python 07_evaluate_design.py --images *.png --countries Japan Singapore
  python 07_evaluate_design.py --images img1.jpg --output /data/output/

Reads from:
  /data/output/design_weights.json  (from Stage 4)

Outputs:
  /data/output/evaluation_report.csv
  /data/output/evaluation_report.png
"""

import os
import sys
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

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR   = Path(os.getenv("DATA_DIR", "/data"))
OUTPUT_DIR = DATA_DIR / "output"
WEIGHTS_JSON = OUTPUT_DIR / "design_weights.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("evaluate_design")

# Ordinal encoding maps (must match 05_cross_analysis.py exactly)
ORNAMENTATION_MAP = {
    "plain": 0.0, "minimal": 0.25, "moderate": 0.5,
    "ornate": 0.75, "highly_ornate": 1.0,
}
AESTHETIC_MAP = {"low": 0.0, "medium": 0.5, "high": 1.0}

DESIGN_ATTRIBUTES = [
    "ornamentation_level",
    "cultural_elements",
    "aesthetic_appeal",
    "motif_diversity",
]

ATTRIBUTE_LABELS = {
    "ornamentation_level": "Ornamentation",
    "cultural_elements":   "Cultural Elements",
    "aesthetic_appeal":    "Aesthetic Appeal",
    "motif_diversity":     "Motif Diversity",
}


# ── Attribute encoding ───────────────────────────────────────────────────────

def encode_attributes(vlm_result: dict) -> dict:
    """Encode VLM categorical outputs to numeric scores (0-1).

    Mirrors 05_cross_analysis.py _encode_image_attributes().
    """
    scores = {}

    # Ornamentation: ordinal string → 0-1
    orn = str(vlm_result.get("ornamentation_level", "minimal")).lower().strip()
    scores["ornamentation_level"] = ORNAMENTATION_MAP.get(orn, 0.25)

    # Cultural elements: boolean → 0 / 1
    ce = vlm_result.get("cultural_elements", False)
    scores["cultural_elements"] = (
        1.0 if str(ce).lower() in ("true", "yes", "1") else 0.0
    )

    # Aesthetic appeal: ordinal string → 0-1
    aa = str(vlm_result.get("aesthetic_appeal", "medium")).lower().strip()
    scores["aesthetic_appeal"] = AESTHETIC_MAP.get(aa, 0.5)

    # Motif diversity: count of unique motif types / 5, capped at 1.0
    motifs_raw = vlm_result.get("motifs", [])
    if isinstance(motifs_raw, str):
        motifs_raw = [m.strip() for m in motifs_raw.replace("|", ",").split(",") if m.strip()]
    motif_list = [m for m in motifs_raw if m and m.lower() != "none"]
    scores["motif_diversity"] = min(len(motif_list) / 5.0, 1.0)

    return scores


def compute_weighted_score(attr_scores: dict, weights: dict) -> float:
    """predicted_sentiment = sum(score[attr] * weight)."""
    score = sum(attr_scores.get(attr, 0) * weights.get(attr, 0)
                for attr in DESIGN_ATTRIBUTES)
    return max(0.0, min(1.0, score))


def predict_label(score: float) -> str:
    if score >= 0.55:
        return "Positive"
    elif score < 0.35:
        return "Negative"
    return "Neutral"


# ── Recommendation engine ────────────────────────────────────────────────────

RECOMMENDATIONS = {
    "ornamentation_level": (
        "Increase pattern complexity — add borders, geometric bands, "
        "or radial motifs to raise visual richness."
    ),
    "cultural_elements": (
        "Add local landmarks, regional symbols, or heritage motifs to "
        "strengthen cultural resonance and public sentiment."
    ),
    "aesthetic_appeal": (
        "Improve color contrast, symmetry, and visual balance — "
        "consider adding decorative borders or central emblems."
    ),
    "motif_diversity": (
        "Incorporate additional design elements — combine floral, "
        "geometric, and text motifs for richer visual storytelling."
    ),
}

WEAKNESS_LABELS = {
    "ornamentation_level": "Ornamentation level is below benchmark",
    "cultural_elements":   "No cultural elements detected",
    "aesthetic_appeal":    "Aesthetic appeal below threshold",
    "motif_diversity":     "Low motif diversity — design too plain",
}


def generate_recommendation(attr_scores: dict, weights: dict):
    """Identify weakest weighted contribution and return (weakness, recommendation)."""
    contributions = {
        attr: attr_scores.get(attr, 0) * weights.get(attr, 0)
        for attr in DESIGN_ATTRIBUTES
    }
    weakest = min(contributions, key=contributions.get)
    weakness = WEAKNESS_LABELS.get(weakest, "Below benchmark")
    rec = RECOMMENDATIONS.get(weakest, "Consider enhancing the design.")
    weak_val = attr_scores.get(weakest, 0)
    rec = f"({ATTRIBUTE_LABELS[weakest]}: {weak_val:.2f}) {rec}"
    return weakness, rec


# ── VLM analysis (reuses Stage 3 infrastructure) ─────────────────────────────

def analyze_with_vlm(image_paths: list) -> list:
    """Run full VLM+LLM pipeline on images using Stage 3 infrastructure.

    Returns list of dicts with VLM analysis results.
    """
    # Import Stage 3 functions
    from importlib import import_module
    stage3 = import_module("04_image_processing")

    client = stage3._get_client()
    cache_file = stage3.CACHE_FILE
    cache = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
            log.info(f"Loaded VLM cache with {len(cache)} entries")
        except Exception:
            cache = {}

    results = []
    for i, img_path in enumerate(image_paths, 1):
        img_path = Path(img_path)
        if not img_path.exists():
            log.warning(f"Image not found: {img_path}")
            results.append({"error": f"File not found: {img_path}", "filename": img_path.name})
            continue

        print(f"[{i}/{len(image_paths)}] Analyzing: {img_path.name}")
        try:
            ai = stage3.analyze_image_ai(client, img_path, cache)
            ai["filename"] = img_path.name
            ai["image_path"] = str(img_path)
            results.append(ai)
            print(f"    ✓ ornamentation={ai.get('ornamentation_level')}  "
                  f"cultural={ai.get('cultural_elements')}  "
                  f"aesthetic={ai.get('aesthetic_appeal')}  "
                  f"motifs={ai.get('motifs', [])}")
        except Exception as e:
            log.error(f"    VLM analysis failed: {e}")
            results.append({"error": str(e), "filename": img_path.name,
                            "image_path": str(img_path)})
    return results


# ── Visualization ─────────────────────────────────────────────────────────────

def generate_report_chart(results: list, weights: dict, output_path: Path):
    """Generate evaluation_report.png with horizontal bar charts + benchmark lines."""
    n = len(results)
    if n == 0:
        print("No results to plot.")
        return

    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n + 1))
    if n == 1:
        axes = [axes]

    colors_map = {
        "Positive": "#3fb950",
        "Neutral":  "#d29922",
        "Negative": "#f85149",
    }

    for ax, r in zip(axes, results):
        name = r.get("filename", "Design")
        score = r.get("score", 0)
        label = r.get("predicted_label", "N/A")

        attrs = DESIGN_ATTRIBUTES
        labels = [ATTRIBUTE_LABELS[a] for a in attrs]
        vals = [r.get("attribute_scores", {}).get(a, 0) for a in attrs]

        # Color bars by value
        bar_colors = ["#3fb950" if v >= 0.6 else "#d29922" if v >= 0.35 else "#f85149"
                      for v in vals]

        y = np.arange(len(labels))
        ax.barh(y, vals, color=bar_colors, edgecolor="#30363d", height=0.55)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10, color="#c9d1d9")
        ax.set_xlim(0, 1.05)

        # Value labels
        for i, v in enumerate(vals):
            ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9, color="#c9d1d9")

        # Overall score indicator
        pred_color = colors_map.get(label, "#8b949e")
        ax.set_title(
            f"{name}  —  Score: {score:.2f} ({label})",
            fontsize=12, fontweight="bold", color="#e6edf3",
        )
        ax.set_facecolor("#0d1117")

        # Benchmark lines
        benchmarks = r.get("benchmark_comparison", [])
        bench_colors = ["#58a6ff", "#bc8cff", "#f0883e", "#3fb950", "#f85149"]
        for bi, bc in enumerate(benchmarks):
            color = bench_colors[bi % len(bench_colors)]
            ax.axvline(bc["benchmark_score"], color=color, linestyle="--",
                       linewidth=1.5, label=f'{bc["country"]} (score: {bc["benchmark_score"]:.2f})')

        # Style
        ax.tick_params(colors="#8b949e")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color("#30363d")

        if benchmarks:
            legend = ax.legend(fontsize=8, loc="lower right",
                               facecolor="#161b22", edgecolor="#30363d")
            for t in legend.get_texts():
                t.set_color("#c9d1d9")

    fig.patch.set_facecolor("#161b22")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: {output_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def evaluate_designs(image_paths: list, comparison_countries: list = None,
                     output_dir: Path = None) -> pd.DataFrame:
    """Evaluate design images against the correlation model.

    Args:
        image_paths: List of image file paths to evaluate.
        comparison_countries: Optional list of country names for benchmarking.
                             If None, uses all countries in the model.
        output_dir: Output directory for report files.

    Returns:
        DataFrame with evaluation results.
    """
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load correlation model ───────────────────────────────────────────
    if not WEIGHTS_JSON.exists():
        print(f"ERROR: {WEIGHTS_JSON} not found. Run the full pipeline (Stages 1-4) first.")
        return pd.DataFrame()

    weights_data = json.loads(WEIGHTS_JSON.read_text())
    weights = weights_data.get("weights", {})
    benchmarks = weights_data.get("country_benchmarks", {})
    correlations = weights_data.get("correlations", {})

    print(f"\n{'='*60}")
    print(f"  Design Evaluation — Correlation Model")
    print(f"{'='*60}")
    print(f"  Model weights:")
    for attr, w in weights.items():
        r = correlations.get(attr, 0)
        print(f"    {attr:25s}  weight={w:.4f}  (r={r:+.4f})")
    print(f"  Countries available: {sorted(benchmarks.keys())}")
    print()

    # Determine benchmark countries
    if comparison_countries:
        # Validate requested countries
        invalid = [c for c in comparison_countries if c not in benchmarks]
        if invalid:
            print(f"  ⚠ Countries not in model (skipping): {invalid}")
        comparison_countries = [c for c in comparison_countries if c in benchmarks]
    else:
        comparison_countries = sorted(benchmarks.keys())

    # ── 2. Validate images ──────────────────────────────────────────────────
    valid_paths = []
    for p in image_paths:
        p = Path(p)
        if p.exists() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
            valid_paths.append(p)
        else:
            print(f"  ⚠ Skipping invalid path: {p}")

    if not valid_paths:
        print("ERROR: No valid image paths provided.")
        return pd.DataFrame()

    print(f"  Evaluating {len(valid_paths)} image(s) against {len(comparison_countries)} benchmark(s)\n")

    # ── 3. Run VLM analysis ─────────────────────────────────────────────────
    vlm_results = analyze_with_vlm(valid_paths)

    # ── 4. Score and benchmark ──────────────────────────────────────────────
    results = []
    for vlm in vlm_results:
        row = {
            "filename": vlm.get("filename", "unknown"),
            "image_path": vlm.get("image_path", ""),
        }

        if "error" in vlm:
            row.update({
                "error": vlm["error"],
                "score": 0, "predicted_label": "Error",
                "ornamentation_level": 0, "cultural_elements": 0,
                "aesthetic_appeal": 0, "motif_diversity": 0,
                "weakness": vlm["error"],
                "recommendation": "Fix VLM error and retry.",
                "benchmark_comparison": [],
                "caption": "",
                "vlm_confidence": 0,
            })
            results.append(row)
            continue

        # Encode attributes
        attr_scores = encode_attributes(vlm)
        score = compute_weighted_score(attr_scores, weights)
        label = predict_label(score)

        # Benchmark comparison
        bench_comparison = []
        for country in comparison_countries:
            bm = benchmarks[country]
            bm_score = sum(
                bm.get(a, 0) * weights.get(a, 0) for a in DESIGN_ATTRIBUTES
            )
            bench_comparison.append({
                "country": country,
                "benchmark_score": round(bm_score, 4),
                "gap": round(score - bm_score, 4),
            })

        weakness, recommendation = generate_recommendation(attr_scores, weights)

        row.update({
            "error": "",
            "score": round(score, 4),
            "predicted_label": label,
            "attribute_scores": {k: round(v, 4) for k, v in attr_scores.items()},
            **{k: round(v, 4) for k, v in attr_scores.items()},
            "weakness": weakness,
            "recommendation": recommendation,
            "caption": vlm.get("caption", ""),
            "vlm_confidence": vlm.get("vlm_confidence", vlm.get("confidence", 0)),
            "ornamentation_raw": vlm.get("ornamentation_level", ""),
            "aesthetic_raw": vlm.get("aesthetic_appeal", ""),
            "cultural_raw": str(vlm.get("cultural_elements", "")),
            "motifs_raw": "|".join(vlm.get("motifs", [])),
            "benchmark_comparison": bench_comparison,
        })
        results.append(row)

    # ── 5. Build output DataFrame ───────────────────────────────────────────
    # Flatten benchmark comparisons into columns
    flat_rows = []
    for r in results:
        flat = {k: v for k, v in r.items() if k != "benchmark_comparison"}
        for bc in r.get("benchmark_comparison", []):
            flat[f"bench_{bc['country']}_score"] = bc["benchmark_score"]
            flat[f"bench_{bc['country']}_gap"] = bc["gap"]
        flat_rows.append(flat)

    df = pd.DataFrame(flat_rows)

    # Save CSV
    csv_path = output_dir / "evaluation_report.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  ✓ Saved: {csv_path}")

    # ── 6. Generate visualization ───────────────────────────────────────────
    png_path = output_dir / "evaluation_report.png"
    generate_report_chart(results, weights, png_path)

    # ── 7. Print summary table ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Evaluation Summary")
    print(f"{'='*60}")
    print(f"  {'Design':25s} {'Score':>7s} {'Label':>10s}  Benchmarks")
    print(f"  {'─'*25} {'─'*7} {'─'*10}  {'─'*30}")
    for r in results:
        name = r["filename"][:25]
        score = r["score"]
        label = r.get("predicted_label", "?")
        bench_str = ""
        for bc in r.get("benchmark_comparison", []):
            gap = bc["gap"]
            sign = "+" if gap >= 0 else ""
            bench_str += f"{bc['country']}({sign}{gap:.2f}) "
        print(f"  {name:25s} {score:>7.3f} {label:>10s}  {bench_str}")

    if any(r.get("weakness") for r in results):
        print(f"\n  ── Recommendations ──")
        for r in results:
            if r.get("weakness"):
                print(f"  {r['filename'][:25]}:")
                print(f"    ⚠ {r['weakness']}")
                print(f"    💡 {r['recommendation']}")

    print(f"\n  Output CSV: {csv_path}")
    print(f"  Output PNG: {png_path}")
    print(f"{'='*60}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate candidate designs against the design–sentiment correlation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 07_evaluate_design.py --images design_a.png design_b.jpg
  python 07_evaluate_design.py --images *.png --countries Japan Singapore
  python 07_evaluate_design.py --images img1.jpg --output /data/output/
        """,
    )
    parser.add_argument(
        "--images", nargs="+", required=True,
        help="Image file paths to evaluate",
    )
    parser.add_argument(
        "--countries", nargs="*", default=None,
        help="Benchmark countries to compare against (default: all in model)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: /data/output/)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    evaluate_designs(
        image_paths=args.images,
        comparison_countries=args.countries,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()