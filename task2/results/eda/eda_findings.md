# Task 2 EDA findings

## Data quality

- The supplied metadata contains 38,617 rows and the image directory contains 38,612 JPEG files.
- 20 rows have no season label and cannot be supervised training examples.
- 5 metadata rows have no matching image.
- 38,592 rows have both a valid season and a matching image.
- The prediction template has 5,829 rows matching 5,829 test images, with the required column order.
- 0 supplied images failed Pillow verification.
- The most common image size is 60x80 (38,595 images).

## Class distribution

- Spring: 1,566 images (4.058%)
- Summer: 19,135 images (49.583%)
- Fall: 10,512 images (27.239%)
- Winter: 7,379 images (19.121%)

The majority-class baseline predicts Summer and obtains 0.4958 accuracy but only 0.1657 macro-F1. The gap occurs because the baseline never recognises the minority seasons.

## Modelling implications

- Use stratified development/holdout splitting and stratified cross-validation so every season is represented proportionally.
- Select GridSearchCV candidates by macro-F1, then report accuracy and weighted-F1 as supporting metrics.
- Tune Task2CNN channel widths, dropout, and learning rate, and use early stopping to control overfitting.
- Report per-class recall and a confusion matrix; overall accuracy can hide failure on Spring.
- Apply one deterministic image-resizing and normalization policy to training, holdout, and test images.
- Season can be a merchandising label rather than an unambiguous visual property, so inspect errors and discuss visually plausible cross-season confusion.
