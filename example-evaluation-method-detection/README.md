# Evaluation method: detection

This is the code that scores the detection phase. It reads algorithm predictions from
`/input/predictions.json`, derives each scan's true lesion presence from the
ground-truth mask (a scan is lesion-positive when its mask contains a connected
component of at least 10 voxels), and writes Sensitivity, Specificity, and Balanced
accuracy to `metrics.json`.

Ground truth is **not** in this repository. It is supplied to the container at runtime
by Grand Challenge, extracted to `/opt/ml/input/data/ground_truth`. This folder
contains only the evaluation code, so you can see exactly how submissions are scored.
