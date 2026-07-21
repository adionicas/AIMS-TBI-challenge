# Evaluation method: segmentation

This is the code that scores the segmentation phase. It reads algorithm predictions
from `/input/predictions.json`, compares each predicted mask to the ground-truth mask,
removes connected components of 50 voxels or fewer from both, and writes the Dice
similarity coefficient (overall, lesion, and no-lesion), the 95th-percentile Hausdorff
distance, and the average symmetric surface distance to `metrics.json`.

Ground truth is **not** in this repository. It is supplied to the container at runtime
by Grand Challenge, extracted to `/opt/ml/input/data/ground_truth`. This folder
contains only the evaluation code, so you can see exactly how submissions are scored.
