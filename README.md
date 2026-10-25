# Anomaly Detection in Network Traffic (NSL-KDD)

This project compares four unsupervised anomaly detectors for network intrusion detection. Each detector learns only from **normal traffic**, so it needs no attack labels and can, in principle, flag attacks it has never seen.

| Detector | Idea | Anomaly score |
| --- | --- | --- |
| Isolation Forest | Anomalies are isolated in fewer random splits | Shorter average path length |
| One-Class SVM | Learns a boundary around normal traffic | Distance outside the boundary |
| Local Outlier Factor | Anomalies sit in sparser neighbourhoods | Local density ratio |
| Autoencoder (32-8-32) | Compresses and rebuilds normal traffic | Reconstruction error |

## Method

1. **Data.** NSL-KDD has 41 flow features. Attack types are grouped into DoS, Probe, R2L and U2R. The test set includes about 17 attack types that never appear in training, which serve as stand-ins for zero-day attacks.
2. **Preprocessing.** Categorical features (protocol, service, flag) are one-hot encoded. Byte counts and duration are log-scaled, and all features are min-max scaled. The preprocessor is fitted on normal training traffic only, so no test or attack data leaks in.
3. **Training.** Each detector is fitted on 80% of the normal training flows.
4. **Threshold.** The threshold is the 95th percentile of scores on the remaining 20% of normal flows, which targets about a 5% false-positive rate. No attack labels are used to choose it.
5. **Evaluation.** Precision, recall, F1, false-positive rate, ROC-AUC and PR-AUC on the test set, plus recall for each attack category, for each attack type, and for unseen versus seen attack types.

## Setup

```bash
pip install -r requirements.txt
```

Download `KDDTrain+.txt` and `KDDTest+.txt` from the [University of New Brunswick NSL-KDD page](https://www.unb.ca/cic/datasets/nsl.html) and place them in `data/`.

## Usage

```bash
python main.py                          # all four detectors on NSL-KDD
python main.py --models iforest autoencoder
python main.py --percentile 99          # stricter threshold, ~1% false-positive target
python main.py --demo                   # synthetic data, checks the pipeline in seconds
```

## Outputs (in `results/`)

| File | Contents |
| --- | --- |
| `metrics.csv` | Overall metrics, threshold and run time per detector |
| `recall_by_category.csv` | Detection rate for DoS, Probe, R2L and U2R |
| `recall_by_attack_type.csv` | Detection rate for each attack type, marked seen or unseen in training |
| `roc_pr_curves.png` | ROC and precision-recall curves for all detectors |
| `recall_by_category.png` | Bar chart of detection rate by category |
| `run_config.json` | Settings used and the list of unseen attack types |

## Project structure

```
main.py            pipeline: load, preprocess, train, threshold, evaluate
src/data.py        NSL-KDD loading, attack categories, preprocessing
src/models.py      the four detectors behind one fit/score interface
src/evaluate.py    metrics and plots
```

## Limitations

NSL-KDD is derived from 1998 traffic and does not reflect modern networks or encrypted traffic. Results show how the methods compare with each other, not the detection rates to expect in deployment. Expect R2L and U2R attacks, which look similar to normal sessions, to be detected far less often than DoS and Probe attacks.

## Author

Firaol Fikadu Disasa, B.Sc. Computer Science final-year project, Hawassa University
