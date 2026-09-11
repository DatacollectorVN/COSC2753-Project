# Task 2 Report Evidence

Task 2 is a four-class, single-label image-classification problem that predicts
`season`. The raw metadata contained 38,617 rows. Twenty rows lacked a season
label and five lacked a matching image, leaving 38,592 usable examples.

The labels are imbalanced: Summer 19,135 (49.583%), Fall 10,512 (27.239%),
Winter 7,379 (19.121%), and Spring 1,566 (4.058%). The majority baseline reaches
0.4958 accuracy but only 0.1657 macro-F1. GridSearchCV therefore selects by
macro-F1 so each season contributes equally to the selection score.

## Preprocessing and Task2CNN

Each image is converted to RGB, padded to preserve aspect ratio, resized to
60 × 80, converted to a channel-first tensor, and normalized to `[-1, 1]`.
Training augmentation applies random horizontal flips and mild brightness and
contrast changes. The supplied files are only read; preprocessing occurs in
memory as batches are loaded.

Task2CNN contains three Conv2d–BatchNorm–ReLU–MaxPool blocks, fixed average
pooling, and a 128-unit classification head. Its 339,876 parameters are
initialized randomly and learned only from the supplied dataset. Training uses
AdamW, class-aware cross-entropy, dropout, weight decay, learning-rate
reduction, and early stopping.

## Evaluation design

Seed 42 creates a stratified split of 30,873 development rows and 7,719 untouched
holdout rows. Three-fold GridSearchCV operates only on the development set.
Each CNN fit uses an internal stratified 10% validation split for early
stopping. The holdout is evaluated once after parameter selection. A separate
model using the selected settings is then fitted on all 38,592 usable rows for
test prediction.

## GridSearchCV result

| Rank | Channels | Dropout | Mean CV macro-F1 | CV standard deviation | Mean CV accuracy |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | [32, 64, 128] | 0.25 | 0.7155 | 0.0075 | 0.7179 |
| 2 | [24, 48, 96] | 0.25 | 0.7130 | 0.0053 | 0.7132 |
| 3 | [24, 48, 96] | 0.40 | 0.7087 | 0.0028 | 0.7082 |
| 4 | [32, 64, 128] | 0.40 | 0.6971 | 0.0162 | 0.6984 |

All candidates use learning rate 0.001. GridSearchCV selected channels
`[32, 64, 128]` and dropout `0.25` using mean CV macro-F1.

## Holdout result

| Metric | Score |
| --- | ---: |
| Accuracy | 0.7257 |
| Macro-F1 | 0.7249 |
| Weighted-F1 | 0.7250 |
| Balanced accuracy | 0.7029 |

| Season | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Spring | 0.9393 | 0.6422 | 0.7628 | 313 |
| Summer | 0.7524 | 0.7685 | 0.7603 | 3,827 |
| Fall | 0.7074 | 0.6253 | 0.6638 | 2,103 |
| Winter | 0.6592 | 0.7757 | 0.7127 | 1,476 |

The measured holdout accuracy is 72.57%, below the requested 80% target.
Spring has high precision but lower recall, while Fall has the lowest class F1.
This supports reporting macro-F1 and per-class measures alongside accuracy.

The selected development fit ran for 28 epochs and restored its best internal
validation checkpoint. The final all-data fit ran for 30 epochs. The two epoch
figures show training and validation accuracy and error; both legends appear
below the graph. No model-comparison visualization is part of Task 2.

## Evidence files

- EDA findings: `results/eda/eda_findings.md`
- Full CV table: `artifacts/grid_search_results.csv`
- Training summary: `artifacts/latest_train_run.json`
- Accuracy graph: `docs/figures/task2_cnn_epoch_accuracy.png`
- Error graph: `docs/figures/task2_cnn_epoch_error.png`
- Epoch data: `docs/task2_cnn_epoch_history.csv`
- Holdout metrics: `results/train/20260911_104332/metrics.json`
- Classification report: `results/train/20260911_104332/classification_report.csv`
- Confusion matrix: `results/train/20260911_104332/confusion_matrix.png`
- Exact split IDs: `results/train/20260911_104332/split_manifest.json`
- Final trained model: `artifacts/season_model.joblib`
- Final predictions: `task2_predictions.csv`
