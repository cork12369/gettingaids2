"""
STAGE 6: Confusion Matrix — Human vs AI Sentiment Validation
Compares human grades against AI sentiment predictions.

Reads from:
  /data/grade_sample.csv      (fixed 2000-snippet sample)
  /data/human_grades.csv      (human scores: 0/1/2)
  /data/text_with_sentiment.csv (AI predictions)

Outputs:
  /data/output/confusion_matrix.png          all plots in a single figure
  /data/output/classification_report.csv     precision/recall/F1 per class
  /data/output/country_accuracy.csv          per-country accuracy scores
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, cohen_kappa_score, accuracy_score
)

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("/data")
OUTPUT_DIR = DATA_DIR / "output"
GRADE_SAMPLE = DATA_DIR / "grade_sample.csv"
HUMAN_GRADES = DATA_DIR / "human_grades.csv"
TEXT_SENTIMENT = DATA_DIR / "text_with_sentiment.csv"

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
# AI sentiment uses -1/0/1; human uses 0/1/2
AI_TO_HUMAN = {-1: 0, 0: 1, 1: 2}


def load_and_merge():
    """Merge human grades with AI predictions on snippet_id."""
    if not GRADE_SAMPLE.exists():
        print("ERROR: grade_sample.csv not found. Open /grade first to generate the sample.")
        return None

    if not HUMAN_GRADES.exists():
        print("ERROR: human_grades.csv not found. No grading has been done yet.")
        return None

    sample = pd.read_csv(GRADE_SAMPLE)
    grades = pd.read_csv(HUMAN_GRADES)

    if len(grades) == 0:
        print("ERROR: No human grades recorded yet.")
        return None

    # Take the first grade per snippet (if multiple graders graded the same one)
    grades_dedup = grades.drop_duplicates(subset="snippet_id", keep="first")

    # Merge
    merged = sample.merge(grades_dedup[["snippet_id", "human_score"]], on="snippet_id", how="inner")

    if len(merged) == 0:
        print("ERROR: No matching snippets between sample and grades.")
        return None

    # Convert AI sentiment to human scale (0/1/2)
    if "sentiment" in merged.columns:
        merged["ai_score"] = merged["sentiment"].map(AI_TO_HUMAN)
    elif "label" in merged.columns:
        label_map = {
            "negative": 0, "NEGATIVE": 0, "LABEL_0": 0,
            "neutral": 1, "NEUTRAL": 1, "LABEL_1": 1,
            "positive": 2, "POSITIVE": 2, "LABEL_2": 2,
        }
        merged["ai_score"] = merged["label"].map(label_map)
    else:
        print("ERROR: No AI sentiment column found in sample data.")
        return None

    # Convert human_score to int
    merged["human_score"] = merged["human_score"].astype(int)
    merged["ai_score"] = merged["ai_score"].astype(int)

    print(f"Merged dataset: {len(merged)} graded snippets")
    return merged


def compute_metrics(merged):
    """Compute all confusion matrix metrics."""
    y_true = merged["human_score"]
    y_pred = merged["ai_score"]

    # Raw confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Normalized confusion matrix (by row = by true label)
    cm_norm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize="true")

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # Per-class metrics
    report = classification_report(y_true, y_pred, labels=[0, 1, 2],
                                   target_names=["Negative", "Neutral", "Positive"],
                                   output_dict=True, zero_division=0)

    # Per-country accuracy
    country_acc = {}
    if "country" in merged.columns:
        for country in sorted(merged["country"].unique()):
            sub = merged[merged["country"] == country]
            if len(sub) >= 3:
                country_acc[country] = round(accuracy_score(sub["human_score"], sub["ai_score"]), 3)

    return {
        "confusion_matrix": cm,
        "confusion_matrix_norm": cm_norm,
        "accuracy": accuracy,
        "kappa": kappa,
        "report": report,
        "country_accuracy": country_acc,
    }


def plot_all(metrics, merged, output_dir):
    """Generate a comprehensive confusion matrix figure."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ── 1. Raw Confusion Matrix ───────────────────────────────────────────
    ax1 = axes[0, 0]
    cm = metrics["confusion_matrix"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg", "Neu", "Pos"],
                yticklabels=["Neg", "Neu", "Pos"], ax=ax1, linewidths=0.5)
    ax1.set_title("Raw Confusion Matrix\n(Human rows × AI columns)", fontweight="bold", fontsize=12)
    ax1.set_xlabel("AI Prediction", fontsize=11)
    ax1.set_ylabel("Human Score", fontsize=11)

    # ── 2. Normalized Confusion Matrix ────────────────────────────────────
    ax2 = axes[0, 1]
    cm_norm = metrics["confusion_matrix_norm"]
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="RdYlGn", xticklabels=["Neg", "Neu", "Pos"],
                yticklabels=["Neg", "Neu", "Pos"], ax=ax2, linewidths=0.5,
                vmin=0, vmax=1)
    ax2.set_title("Normalized Confusion Matrix\n(% of each human row)", fontweight="bold", fontsize=12)
    ax2.set_xlabel("AI Prediction", fontsize=11)
    ax2.set_ylabel("Human Score", fontsize=11)

    # ── 3. Per-Class Metrics Bar Chart ────────────────────────────────────
    ax3 = axes[1, 0]
    report = metrics["report"]
    classes = ["Negative", "Neutral", "Positive"]
    precision = [report[c]["precision"] for c in classes]
    recall = [report[c]["recall"] for c in classes]
    f1 = [report[c]["f1-score"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25
    ax3.bar(x - width, precision, width, label="Precision", color="#3498db")
    ax3.bar(x, recall, width, label="Recall", color="#2ecc71")
    ax3.bar(x + width, f1, width, label="F1", color="#e74c3c")
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes)
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("Score", fontsize=11)
    ax3.set_title("Per-Class Precision / Recall / F1", fontweight="bold", fontsize=12)
    ax3.legend()
    ax3.axhline(0.5, color="gray", linestyle="--", alpha=0.3)

    # ── 4. Per-Country Accuracy ───────────────────────────────────────────
    ax4 = axes[1, 1]
    country_acc = metrics["country_accuracy"]
    if country_acc:
        countries = list(country_acc.keys())
        accs = list(country_acc.values())
        colors = ["#2ecc71" if a >= 0.6 else "#f39c12" if a >= 0.4 else "#e74c3c" for a in accs]
        bars = ax4.barh(countries, accs, color=colors, edgecolor="black", linewidth=0.5)
        ax4.axvline(metrics["accuracy"], color="#58a6ff", linewidth=2, linestyle="--",
                    label=f'Overall: {metrics["accuracy"]:.1%}')
        ax4.set_xlim(0, 1.05)
        ax4.set_xlabel("Accuracy", fontsize=11)
        ax4.set_title("Per-Country Accuracy", fontweight="bold", fontsize=12)
        ax4.legend(loc="lower right")
        for bar, val in zip(bars, accs):
            ax4.text(val + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.1%}",
                     va="center", fontsize=9)
    else:
        ax4.text(0.5, 0.5, "No country data", ha="center", va="center",
                 transform=ax4.transAxes, fontsize=14, color="#8b949e")
        ax4.set_title("Per-Country Accuracy", fontweight="bold", fontsize=12)

    # ── Super-title with headline metrics ─────────────────────────────────
    n = len(merged)
    fig.suptitle(
        f"Human-AI Sentiment Validation\n"
        f"N={n} snippets  |  Accuracy={metrics['accuracy']:.1%}  |  "
        f"Cohen's κ={metrics['kappa']:.3f}",
        fontsize=14, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: confusion_matrix.png")


def save_reports(metrics, output_dir):
    """Save CSV reports."""
    # Classification report
    report = metrics["report"]
    rows = []
    for cls in ["Negative", "Neutral", "Positive"]:
        rows.append({
            "class": cls,
            "precision": round(report[cls]["precision"], 3),
            "recall": round(report[cls]["recall"], 3),
            "f1_score": round(report[cls]["f1-score"], 3),
            "support": report[cls]["support"],
        })
    rows.append({
        "class": "OVERALL",
        "precision": round(report["macro avg"]["precision"], 3),
        "recall": round(report["macro avg"]["recall"], 3),
        "f1_score": round(report["macro avg"]["f1-score"], 3),
        "support": report["macro avg"]["support"],
    })
    report_df = pd.DataFrame(rows)
    report_df.to_csv(output_dir / "classification_report.csv", index=False)
    print(f"  ✓ Saved: classification_report.csv")

    # Country accuracy
    if metrics["country_accuracy"]:
        ca_df = pd.DataFrame([
            {"country": k, "accuracy": v, "snippets": "varies"}
            for k, v in metrics["country_accuracy"].items()
        ])
        ca_df.to_csv(output_dir / "country_accuracy.csv", index=False)
        print(f"  ✓ Saved: country_accuracy.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and merging data...")
    merged = load_and_merge()
    if merged is None:
        return

    print(f"Computing metrics on {len(merged)} graded snippets...")
    metrics = compute_metrics(merged)

    # Print summary
    print(f"\n{'='*50}")
    print(f"CONFUSION MATRIX RESULTS")
    print(f"{'='*50}")
    print(f"  Snippets graded:  {len(merged)}")
    print(f"  Overall Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Cohen's Kappa:    {metrics['kappa']:.3f}", end="")
    if metrics['kappa'] >= 0.8:
        print("  (Almost perfect agreement)")
    elif metrics['kappa'] >= 0.6:
        print("  (Substantial agreement)")
    elif metrics['kappa'] >= 0.4:
        print("  (Moderate agreement)")
    else:
        print("  (Fair/slight agreement)")

    print(f"\n  Per-Class:")
    for cls in ["Negative", "Neutral", "Positive"]:
        r = metrics["report"][cls]
        print(f"    {cls:>10}: P={r['precision']:.2f}  R={r['recall']:.2f}  F1={r['f1-score']:.2f}  (n={r['support']})")

    if metrics["country_accuracy"]:
        print(f"\n  Per-Country Accuracy:")
        for country, acc in sorted(metrics["country_accuracy"].items()):
            print(f"    {country:>15}: {acc:.1%}")

    # Generate visualizations
    print(f"\nGenerating visualizations...")
    plot_all(metrics, merged, OUTPUT_DIR)

    # Save reports
    save_reports(metrics, OUTPUT_DIR)

    print(f"\n{'='*50}")
    print(f"✓ Confusion Matrix Analysis Complete")
    print(f"{'='*50}")
    print(f"  Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()