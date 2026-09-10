# Task 2 Report Evidence

This file contains evidence and defensible interpretation for the Task 2 section of the assignment report. Condense it to fit the group report's five-page limit. Figures and supporting tables may be placed in the appendix, but the assignment brief does not allow the appendix to contain the judgement itself.

## Problem and data

Task 2 was treated as four-class, single-label classification from image pixels to `season`. The raw metadata contained 38,617 rows. Twenty rows had no season target and five rows had no corresponding image, leaving 38,592 supervised examples. No supplied JPEG failed verification. Most images were 60 × 80 pixels.

The usable labels were imbalanced: Summer 19,135 (49.583%), Fall 10,512 (27.239%), Winter 7,379 (19.121%), and Spring 1,566 (4.058%). A classifier that always predicted Summer obtained 0.4958 holdout accuracy but only 0.1657 macro-F1. Macro-F1 was therefore used for model selection, while accuracy, weighted-F1, balanced accuracy, and per-class metrics were retained as supporting evidence.

## Preprocessing and representation

Every image was converted to RGB, padded when necessary to preserve the dominant 3:4 aspect ratio, and resized to 48 × 64. The reduced size controlled memory and computation during 40 cross-validation fits. The final 1,308-dimensional representation combined HOG edge-orientation descriptors with normalized HSV histograms. HOG represented garment or product shape, while HSV histograms retained global colour information that could contribute to season labelling.

Only pixels were used to construct predictors. Fields such as `articleType`, `baseColour`, and `productDisplayName` were deliberately excluded because the task asks for a prediction from the fashion image.

## Evaluation design

A fixed seed of 42 created a stratified 80/20 split: 30,873 development examples and 7,719 holdout examples. Five-fold stratified GridSearchCV was applied only to the development set. `StandardScaler` remained inside the estimator pipeline, so every fold fitted its scaler using only that fold's training portion. The holdout did not influence preprocessing statistics, fitting, hyperparameter selection, or refitting decisions.

After the holdout evaluation, the selected pipeline was cloned and trained on all 38,592 usable examples. The development-only model was retained separately so the reported holdout metrics remain reproducible and are not accidentally calculated with the all-data model.

## GridSearchCV results

The search compared four regularization strengths with and without automatic inverse-frequency class weighting.

| Rank | C | Class weight | Mean CV macro-F1 | CV standard deviation | Mean CV accuracy |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 0.01 | None | 0.6720 | 0.0115 | 0.6854 |
| 2 | 0.1 | None | 0.6633 | 0.0087 | 0.6814 |
| 3 | 1.0 | None | 0.6572 | 0.0118 | 0.6798 |
| 4 | 10.0 | None | 0.6553 | 0.0127 | 0.6794 |
| 5 | 0.01 | Balanced | 0.6251 | 0.0074 | 0.6605 |
| 6 | 0.1 | Balanced | 0.6141 | 0.0089 | 0.6542 |
| 7 | 1.0 | Balanced | 0.6107 | 0.0085 | 0.6521 |
| 8 | 10.0 | Balanced | 0.6103 | 0.0083 | 0.6519 |

The best result came from the strongest tested regularization (`C=0.01`) without class weighting. Performance declined as `C` increased, indicating that weaker regularization did not generalize as well. Balanced weighting increased the influence of rare examples but reduced macro-F1 for every tested `C`; it was therefore rejected using measured cross-validation evidence rather than assumed to help because the labels were imbalanced.

## Holdout results

| Measure | Majority baseline | Selected HOG/HSV + LinearSVC |
| --- | ---: | ---: |
| Accuracy | 0.4958 | 0.6867 |
| Macro-F1 | 0.1657 | 0.6684 |
| Balanced accuracy | 0.2500 | 0.6299 |
| Weighted-F1 | 0.3287 | 0.6820 |

| Season | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Spring | 0.8743 | 0.5335 | 0.6627 | 313 |
| Summer | 0.6829 | 0.7975 | 0.7358 | 3,827 |
| Fall | 0.6318 | 0.5231 | 0.5723 | 2,103 |
| Winter | 0.7451 | 0.6653 | 0.7029 | 1,476 |

The selected model improved macro-F1 by 0.5027 over the majority baseline and produced useful predictions for all four classes. Summer had the highest recall. Fall recall was the lowest, while Spring showed high precision but moderate recall: predicted Spring examples were usually correct, but the model missed almost half of the Spring items. These class-specific results are more informative than overall accuracy alone.

## Ultimate judgement

The `C=0.01`, unweighted HOG/HSV plus LinearSVC pipeline is the recommended Task 2 model. It was the highest-ranked candidate under the preselected macro-F1 criterion, its holdout macro-F1 was close to its mean cross-validation value, and it substantially exceeded the majority baseline. The saved model is approximately 70 KB and performs CPU inference over the extracted test feature matrix quickly, making it practical to distribute and rerun on a standard machine.

This judgement is limited by the moderate recall for Spring and Fall and by the nature of the target. A merchandising season is not always a visible property: visually similar products may receive different season labels because of release timing or business decisions. The sample images also show cosmetics, accessories, footwear, and clothing within the same target space, so the estimator can learn dataset-specific associations between product shape and season rather than a universal concept of seasonal suitability.

The unlabelled test prediction distribution was Summer 3,403, Winter 1,031, Spring 917, and Fall 478. These counts should be reported as a diagnostic, not as accuracy evidence. In particular, their difference from the training distribution could reflect test-set shift or model error; without test labels, those explanations cannot be separated.

## Independent comparison and literature

Seo, Lee, and Jang (2025) used the same broader Fashion Product Images Dataset for category classification and reported severe class imbalance and difficulties caused by product images that contain models or multiple visible items. They used under-sampling, pretrained ResNet50 image features, BERT title features, and multimodal fusion. Their result is useful evidence that imbalance, visual noise, and non-image information materially affect this dataset. It is not a numerical benchmark for this Task 2 model because their target, input modalities, sampling, image resolution, and pretrained systems differ. Their pretrained and multimodal final method also falls outside this assignment's rule that the submitted model be fully trained on the supplied dataset. Source: [Seo et al., 2025, PLOS One](https://doi.org/10.1371/journal.pone.0324621).

Liu et al. (2016) introduced DeepFashion and trained FashionNet to jointly predict clothing attributes and landmarks, while Jia et al. (2020) introduced Fashionpedia for localized fine-grained fashion attributes. These works support the broader view that fashion understanding benefits from localized and jointly learned visual representations. They use different data, targets, and deep architectures, so their published scores must not be directly compared with the present season-classification score. Sources: [Liu et al., 2016, CVPR](https://openaccess.thecvf.com/content_cvpr_2016/html/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.html) and [Jia et al., 2020, ECCV](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1203_ECCV_2020_paper.php).

For a true independent numerical evaluation, collect a small set of fashion product images outside the supplied split and assign the four seasons using a documented annotation rule. The assignment brief requires any additionally collected or preprocessed dataset to be made accessible to the evaluator. Until such labels exist, the literature comparison should remain qualitative and the stratified holdout should remain the only claimed numerical generalization estimate.

## Evidence files

- EDA summary: `results/eda/eda_findings.md`
- Complete GridSearchCV table: `artifacts/grid_search_results.csv`
- Training summary: `results/train/20260910_230121/scores.json`
- Holdout metrics: `results/train/20260910_230121/metrics.json`
- Classification report: `results/train/20260910_230121/classification_report.csv`
- Confusion matrix: `results/train/20260910_230121/confusion_matrix.png`
- Exact split IDs: `results/train/20260910_230121/split_manifest.json`
- Stable final model: `artifacts/season_model.joblib`
- Final Task 2 predictions: `task2_predictions.csv`
