"""Anomaly detection in network traffic (NSL-KDD).

Trains four unsupervised detectors on normal traffic only, sets each
detector's threshold from held-out normal traffic (no attack labels used),
then evaluates on the NSL-KDD test set, including attack types that never
appear in training.

Usage:
    python main.py --data-dir data/            # real NSL-KDD
    python main.py --demo                      # synthetic data, quick test
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import Preprocessor, load_nsl_kdd, make_synthetic
from src.evaluate import (binary_metrics, plot_category_recall, plot_curves,
                          recall_by_group, threshold_from_normal)
from src.models import ALL_DETECTORS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data", help="folder with KDDTrain+.txt and KDDTest+.txt")
    p.add_argument("--demo", action="store_true", help="run on synthetic data instead")
    p.add_argument("--models", nargs="+", default=list(ALL_DETECTORS), choices=list(ALL_DETECTORS))
    p.add_argument("--percentile", type=float, default=95.0,
                   help="threshold = this percentile of normal validation scores (default 95)")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    if args.demo:
        print("Running on SYNTHETIC data (pipeline test only).")
        train, test = make_synthetic(seed=args.seed)
    else:
        train, test = load_nsl_kdd(args.data_dir)
    unseen_types = sorted(set(test.loc[test.is_attack == 1, "label"]) - set(train["label"]))
    test["unseen"] = test["label"].isin(unseen_types)
    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows "
          f"({test.is_attack.mean():.1%} attacks, {test.unseen.sum():,} from "
          f"{len(unseen_types)} attack types never seen in training)")

    # 2. Normal training traffic only: split into fit and threshold-validation sets
    normal = train[train.is_attack == 0]
    normal_fit, normal_val = train_test_split(normal, test_size=0.2, random_state=args.seed)
    prep = Preprocessor().fit(normal_fit)
    X_fit, X_val, X_test = (prep.transform(d) for d in (normal_fit, normal_val, test))
    y_test = test["is_attack"].to_numpy()
    print(f"Features after encoding: {X_fit.shape[1]} | trained on {len(X_fit):,} normal flows\n")

    # 3. Train, threshold and evaluate each detector
    rows, all_scores, per_category = [], {}, {}
    for key in args.models:
        detector = ALL_DETECTORS[key](seed=args.seed)
        start = time.time()
        detector.fit(X_fit)
        threshold = threshold_from_normal(detector.score(X_val), args.percentile)
        scores = detector.score(X_test)
        elapsed = time.time() - start

        metrics = binary_metrics(y_test, scores, threshold)
        unseen_mask = test["unseen"].to_numpy()
        metrics["recall_unseen_types"] = float((scores[unseen_mask] > threshold).mean()) if unseen_mask.any() else float("nan")
        metrics["recall_seen_types"] = float((scores[(y_test == 1) & ~unseen_mask] > threshold).mean())
        metrics.update(model=detector.name, threshold=threshold, seconds=round(elapsed, 1))
        rows.append(metrics)
        all_scores[detector.name] = scores
        per_category[detector.name] = recall_by_group(test, scores, threshold, "category")
        print(f"{detector.name:<22} F1 {metrics['f1']:.3f} | recall {metrics['recall']:.3f} | "
              f"FPR {metrics['fpr']:.3f} | PR-AUC {metrics['pr_auc']:.3f} | "
              f"unseen recall {metrics['recall_unseen_types']:.3f} | {elapsed:.0f}s")

    # 4. Save results
    results = pd.DataFrame(rows).set_index("model")
    per_category = pd.DataFrame(per_category)
    per_type = pd.DataFrame({name: recall_by_group(test, s, results.loc[name, "threshold"], "label")
                             for name, s in all_scores.items()})
    per_type.insert(0, "unseen_in_training", per_type.index.isin(unseen_types))

    results.round(4).to_csv(out_dir / "metrics.csv")
    per_category.round(4).to_csv(out_dir / "recall_by_category.csv")
    per_type.round(4).to_csv(out_dir / "recall_by_attack_type.csv")
    (out_dir / "run_config.json").write_text(json.dumps(
        {**vars(args), "unseen_attack_types": unseen_types}, indent=2))
    plot_curves(y_test, all_scores, out_dir)
    plot_category_recall(per_category, out_dir)

    print("\nDetection rate by attack category:")
    print(per_category.round(3).to_string())
    print(f"\nResults saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
