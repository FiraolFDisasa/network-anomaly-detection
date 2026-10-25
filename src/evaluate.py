"""Metrics and plots for anomaly detectors."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def threshold_from_normal(normal_scores: np.ndarray, percentile: float) -> float:
    """Label-free threshold: a percentile of scores on held-out normal
    traffic. A 95th percentile targets about a 5% false-positive rate."""
    return float(np.percentile(normal_scores, percentile))


def binary_metrics(y_true, scores, threshold) -> dict:
    y_pred = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def recall_by_group(test: pd.DataFrame, scores, threshold, column: str) -> pd.Series:
    """Detection rate for each attack category (or attack type)."""
    attacks = test["is_attack"] == 1
    flagged = pd.Series(scores > threshold, index=test.index)
    return flagged[attacks].groupby(test.loc[attacks, column]).mean()


def plot_curves(y_true, all_scores: dict, out_dir: Path):
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, scores in all_scores.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_true, scores):.3f})")
        ax_pr.plot(rec, prec, label=f"{name} (AP {average_precision_score(y_true, scores):.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax_roc.set(title="ROC curve", xlabel="False-positive rate", ylabel="True-positive rate")
    ax_pr.axhline(np.mean(y_true), color="k", ls="--", lw=0.8)
    ax_pr.set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
    for ax in (ax_roc, ax_pr):
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr_curves.png", dpi=150)
    plt.close(fig)


def plot_category_recall(per_category: pd.DataFrame, out_dir: Path):
    ax = per_category.plot(kind="bar", figsize=(9, 4.5), rot=0)
    ax.set(title="Detection rate by attack category", ylabel="Recall", xlabel="", ylim=(0, 1.05))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "recall_by_category.png", dpi=150)
    plt.close()
