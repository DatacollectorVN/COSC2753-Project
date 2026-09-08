# Task 1: `articleType` Class-Imbalance Analysis

## Objective

Develop an image-classification model that predicts `articleType` from the FashionDataset product images. The main data-quality issue is the large number of target classes with very few examples.

## Dataset observations

The image EDA and training metadata show:

- 38,617 rows in `styles_train.csv`.
- 38,612 usable training images; five CSV records have no corresponding image.
- 125 `articleType` values appear in the CSV.
- Only 124 classes have a usable image because the sole `Suits` image is missing.
- Class frequency ranges from 6,781 images for `Tshirts` to one image for several rare classes.

The distribution at possible minimum-frequency thresholds is:

| Minimum examples retained | Named classes retained | Classes below threshold | Images below threshold | Dataset share |
|---:|---:|---:|---:|---:|
| 20 | 76 | 48 | 338 | 0.88% |
| 50 | 59 | 65 | 917 | 2.37% |
| 100 | 46 | 78 | 1,813 | 4.70% |

With an 80/20 train-validation split, a class containing 50 images would have approximately 40 training examples and 10 validation examples. This is still a small sample, but it is a reasonable minimum for an initial transfer-learning experiment.

## Image-size decision

The EDA shows that the source images are small and almost completely uniform in size:

| Split | Images | Images at 60 × 80 px | Share at 60 × 80 px |
|---|---:|---:|---:|
| Train | 38,612 | 38,595 | 99.96% |
| Test | 5,829 | 5,823 | 99.90% |

The normal source aspect ratio is therefore **3:4** (`width:height`). The few exceptions range from 53–60 pixels wide and 60–80 pixels high.

### Selected input size: 96 × 128 pixels

Use a model input of **96 pixels wide × 128 pixels high**, represented as `(height, width) = (128, 96)` in most deep-learning APIs.

This size is recommended because it:

- Preserves the original 3:4 aspect ratio, so product shapes are not stretched.
- Upscales each dimension by only 1.6×, avoiding excessive interpolation of the low-resolution originals.
- Gives a four-block CNN a 6 × 8 feature map after four 2× pooling operations, instead of approximately 3 × 5 at the original resolution.
- Uses 12,288 input pixels, approximately 75.5% fewer than a 224 × 224 input, reducing memory use and training time.
- Matches the nearly identical train and test image-size distributions.

The current transform resizes every image to a square because `Resize` accepts one integer and creates `(size, size)`. With `IMAGE_SIZE = 224`, a 60 × 80 image is stretched from 3:4 to 1:1 and expanded to more than ten times its original pixel area. Upscaling cannot recover detail that is absent from the source image.

Change the resize transform to accept separate height and width values. For the small number of nonstandard images, first pad them to a 60 × 80 canvas while preserving their aspect ratio, then resize to 96 × 128. Padding is preferable to stretching or cropping because the product should remain geometrically intact.

Conceptual preprocessing:

```python
INPUT_HEIGHT = 128
INPUT_WIDTH = 96

# 1. Fit the image within a 3:4 canvas without distortion.
# 2. Pad the remaining area using a neutral/background colour.
# 3. Resize the canvas to (width=96, height=128).
image = resize_with_padding(image, width=INPUT_WIDTH, height=INPUT_HEIGHT)
```

Use the same deterministic resize-and-pad operation for training, validation, and test data. Training-only augmentation should run after geometry has been standardized.

### Image-size validation experiment

Although 96 × 128 is the selected default, verify it with a controlled comparison:

| Configuration | Purpose |
|---|---|
| 60 × 80 | Native-resolution efficiency baseline |
| **96 × 128** | Recommended balance of feature-map size and compute |
| 120 × 160 | Check whether moderate additional upscaling helps |
| 224 × 224 square | Existing baseline; measures the cost of distortion and extra compute |

Keep the data split, target mapping, model, optimizer, augmentation, seed, and epoch budget fixed. Compare macro-F1, validation loss, training time per epoch, and peak memory. Retain 96 × 128 unless a larger input produces a meaningful and repeatable macro-F1 improvement.

If a pretrained network strictly expects a square 224 × 224 input, resize the 60 × 80 content to 168 × 224 and pad horizontally to 224 × 224. This preserves the 3:4 product geometry while remaining compatible with the pretrained interface.

## Recommendation

If the project permits redefining the target labels, use **50 images as the initial cutoff**:

- Preserve the 59 `articleType` classes containing at least 50 usable images.
- Map the 65 smaller classes to `Other`.
- Train a classifier with 60 outputs: 59 named classes plus `Other`.
- This preserves a specific `articleType` label for 97.63% of usable images.

Example mapping:

```python
MIN_CLASS_COUNT = 50

counts = df["articleType"].value_counts()
rare_labels = counts[counts < MIN_CLASS_COUNT].index

df["model_target"] = df["articleType"].where(
    ~df["articleType"].isin(rare_labels),
    "Other",
)
```

The cutoff should be treated as a hyperparameter rather than a final assumption. Compare thresholds of 20, 50, and 100 on the same validation partitions.

## Important limitation of `Other`

The rare labels are not visually or semantically homogeneous. A single `Other` class may contain unrelated products such as clothing, cosmetics, bags, electronics, and accessories. Consequently, the model must learn a union of many disconnected visual categories.

Grouping is therefore a practical simplification, not a perfect representation of the target. A stronger future design would be hierarchical classification:

1. Predict `masterCategory` or `subCategory`.
2. Predict an `articleType` valid within that parent category.

This respects the dataset taxonomy and reduces competition among unrelated labels.

## Evaluation constraint

Before grouping labels, confirm the required output schema:

- If grading or hidden-test evaluation expects the original `articleType` names, `Other` is not a valid replacement. In that case, retain the original classes and address imbalance through the training procedure.
- If the project evaluates a taxonomy designed by the team, introducing `Other` is acceptable as long as the transformation is applied consistently to training, validation, inference, and evaluation.

Do not silently remove rare examples. Either preserve their original classes or explicitly map them to a documented fallback label.

## Required modeling changes

### Stratified splitting

The current data loader randomly shuffles all rows before creating training and validation subsets. Rare classes may therefore be absent from one split, making per-class evaluation unstable or impossible.

Use a stratified split based on the final target (`model_target` after grouping). Fix and record the random seed so all experiments use identical partitions.

### Transfer learning

Prefer a pretrained image backbone such as ResNet or EfficientNet over training the current simple CNN entirely from scratch. Transfer learning is especially valuable for classes with tens rather than thousands of examples.

### Imbalance handling

Evaluate one imbalance treatment at a time:

- Class-weighted cross-entropy.
- Weighted or class-aware sampling.
- Focal loss if the weighted-loss baseline remains dominated by common classes.
- Image augmentation applied only to the training split.

Avoid combining every technique in the first experiment because it becomes difficult to identify which change produced an improvement.

## Proposed experiments

Use the same pretrained backbone, preprocessing, stratified partitions, seed, and training budget for each experiment.

| Experiment | Target definition | Purpose |
|---|---|---|
| A | All usable original classes | Establish the ungrouped baseline |
| B | Classes with fewer than 20 images → `Other` | Minimal intervention |
| C | Classes with fewer than 50 images → `Other` | Recommended initial balance |
| D | Classes with fewer than 100 images → `Other` | Test a more conservative class-size requirement |

For every experiment, report:

- Overall accuracy.
- Macro precision, recall, and F1.
- Weighted F1.
- Per-class precision, recall, F1, and support.
- Confusion matrix.
- Results for `Other` separately, when applicable.

Macro-F1 should be the main selection metric because it gives each retained class equal importance. Accuracy and weighted-F1 can conceal failure on smaller classes.

## Decision rule

Adopt the `< 50 → Other` configuration only if it:

1. Improves macro-F1 and validation stability over the all-class baseline.
2. Does not achieve the improvement merely by over-predicting `Other`.
3. Maintains acceptable recall for the named classes.
4. Is compatible with the required test-output labels.

If these conditions are not met, retain the original labels and use transfer learning plus imbalance-aware training, or move to a hierarchical classifier.
