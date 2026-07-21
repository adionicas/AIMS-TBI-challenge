"""
AIMS-TBI 2026 segmentation evaluation method.

Voxel-level lesion segmentation: for each algorithm job, compare the predicted
lesion mask to the ground-truth mask and report the Dice similarity coefficient
(overall, lesion-only, and no-lesion), the 95th-percentile Hausdorff distance, and
the average symmetric surface distance. Connected components smaller than the
segmentation threshold (over 50 voxels) are removed from both masks before scoring.

Runtime contract (Grand Challenge 2026 development kit schema):
  - predictions described in /input/predictions.json (socket schema)
  - predicted mask located under /input/<job-pk>/output/ (located by search, so the
    container does not depend on the exact output socket slug)
  - ground truth uploaded separately as a tarball, extracted to
    /opt/ml/input/data/ground_truth/  (NOT bundled into the image)
"""

import glob
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import cc3d
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, distance_transform_edt

# Grand Challenge runtime directories
INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
GROUND_TRUTH_DIRECTORY = Path("/opt/ml/input/data/ground_truth")

# Input socket carrying the T1-weighted brain MRI (used to match the ground truth)
INPUT_IMAGE_SLUG = "t1-brain-mri"

# Design document (Task 2): segmentation is scored on lesions over 50 voxels.
# Connected components of this size or smaller are removed from both masks first.
MIN_LESION_SIZE_VOXELS = 50
CONNECTIVITY = 26


def main():
    predictions = read_predictions()

    # Show what is actually in the ground-truth directory at runtime.
    log_directory(GROUND_TRUTH_DIRECTORY, label="Ground truth")

    records = []
    for job in predictions:
        image_name = get_image_name(values=job["inputs"], slug=INPUT_IMAGE_SLUG)
        try:
            prediction_path = find_prediction_mask(job["pk"])
            ground_truth_path = find_ground_truth_path(image_name)
        except Exception as e:
            warnings.warn(f"{image_name}: {e}. Skipping.")
            continue

        try:
            gt, pred, spacing = load_aligned(ground_truth_path, prediction_path)
        except Exception as e:
            warnings.warn(f"Could not load masks for {image_name}: {e}. Skipping.")
            continue

        gt = remove_small_components(gt)
        pred = remove_small_components(pred)

        gt_has_lesion = bool(gt.any())
        dice = compute_dice(pred, gt)

        # Surface metrics are only defined when both masks contain something.
        if gt.any() and pred.any():
            hd95_value = hausdorff95(pred, gt, spacing)
            assd_value = assd(pred, gt, spacing)
        else:
            hd95_value = None
            assd_value = None

        records.append({
            "image_name": image_name,
            "ground_truth_has_lesion": int(gt_has_lesion),
            "Dice": float(dice),
            "HD95": hd95_value,
            "ASSD": assd_value,
        })
        print(f"[INFO] {image_name}: gt_lesion={gt_has_lesion}, "
              f"Dice={dice:.4f}, HD95={hd95_value}, ASSD={assd_value}")

    pd.DataFrame(records).to_csv(OUTPUT_DIRECTORY / "individual_metrics.csv", index=False)
    print(f"\n \t | Individual metrics written: {OUTPUT_DIRECTORY / 'individual_metrics.csv'}")

    get_aggregates(records)
    return 0


def read_predictions():
    with open(INPUT_DIRECTORY / "predictions.json") as f:
        return json.load(f)


def log_directory(directory, *, label):
    """Print the contents of a directory so the actual files (in particular the
    uploaded ground-truth tarball) are visible in the evaluation logs."""
    directory = Path(directory)
    print(f"\n[INFO] ===== {label}: {directory} =====")
    if not directory.exists():
        print(f"[WARN] {label} directory does not exist at runtime.")
        return
    files = []
    for root, _dirs, names in os.walk(directory):
        for name in names:
            files.append(os.path.relpath(os.path.join(root, name), directory))
    files.sort()
    ext_counts = {}
    for rel in files:
        low = rel.lower()
        if low.endswith(".nii.gz"):
            key = ".nii.gz"
        elif low.endswith(".nii"):
            key = ".nii"
        elif low.endswith(".mha"):
            key = ".mha"
        else:
            key = "other"
        ext_counts[key] = ext_counts.get(key, 0) + 1
    print(f"[INFO] {label}: {len(files)} file(s); by extension: {ext_counts}")
    shown = files if len(files) <= 200 else files[:200]
    for rel in shown:
        print(f"[INFO]   {rel}")
    if len(files) > len(shown):
        print(f"[INFO]   ... and {len(files) - len(shown)} more")


def get_image_name(*, values, slug):
    for value in values:
        if value["socket"]["slug"] == slug:
            return value["image"]["name"]
    raise RuntimeError(f"Input image for socket '{slug}' not found")


def find_prediction_mask(job_pk):
    """Locate the predicted mask under the job's output, by search, so the container
    does not depend on the exact output socket slug."""
    base = os.path.join(INPUT_DIRECTORY, job_pk, "output")
    for pattern in ("**/*.mha", "**/*.nii.gz", "**/*.nii"):
        hits = sorted(glob.glob(os.path.join(base, pattern), recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No prediction mask found under {base}")


def extract_subject_id(image_name):
    """Subject id from the input image name, e.g. 'scan_0632_T1.nii.gz' -> 'scan_0632'."""
    name = image_name
    for ext in (".nii.gz", ".nii", ".mha"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    lower = name.lower()
    for marker in ("_t1w", "_t1"):
        idx = lower.find(marker)
        if idx != -1:
            return name[:idx]
    return name


def find_ground_truth_path(image_name):
    """Locate the ground-truth mask for an input image by subject id, searching the
    ground-truth directory recursively. Accepts .mha, .nii, or .nii.gz, and either
    'lesion' or 'Lesion' naming, with or without a '_bin' suffix."""
    subject = extract_subject_id(image_name)
    subject_lower = subject.lower()

    candidates = []
    for root, _dirs, files in os.walk(GROUND_TRUTH_DIRECTORY):
        for filename in files:
            low = filename.lower()
            if not (low.endswith(".mha") or low.endswith(".nii.gz") or low.endswith(".nii")):
                continue
            if (subject_lower + "_") in low or (subject_lower + ".") in low:
                candidates.append(os.path.join(root, filename))

    if not candidates:
        raise FileNotFoundError(
            f"No ground-truth mask for subject '{subject}' under {GROUND_TRUTH_DIRECTORY}"
        )

    def preference(path):
        low = os.path.basename(path).lower()
        return (
            0 if "lesion" in low else 1,
            0 if low.endswith(".mha") else 1,
            len(low),
        )

    return sorted(candidates, key=preference)[0]


def load_aligned(ground_truth_path, prediction_path):
    """Load ground truth and prediction. Resample the prediction onto the ground-truth
    grid with nearest-neighbour interpolation if the shapes differ. Returns
    (ground_truth_bool, prediction_bool, spacing) where spacing is in array (z, y, x)
    order in millimetres."""
    gt_img = sitk.ReadImage(str(ground_truth_path))
    pred_img = sitk.ReadImage(str(prediction_path))

    gt = sitk.GetArrayFromImage(gt_img)
    pred = sitk.GetArrayFromImage(pred_img)

    if gt.shape != pred.shape:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(gt_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        pred_img = resampler.Execute(pred_img)
        pred = sitk.GetArrayFromImage(pred_img)

    sx, sy, sz = gt_img.GetSpacing()  # SimpleITK gives (x, y, z)
    spacing = (sz, sy, sx)            # arrays are (z, y, x)
    return gt.astype(bool), pred.astype(bool), spacing


def remove_small_components(mask, min_size=MIN_LESION_SIZE_VOXELS, connectivity=CONNECTIVITY):
    """Remove connected components of `min_size` voxels or fewer, keeping lesions over
    the threshold."""
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return mask
    labels, n = cc3d.connected_components(mask, connectivity=connectivity, return_N=True)
    if n == 0:
        return mask
    counts = np.bincount(labels.reshape(-1))
    counts[0] = 0  # background
    keep_labels = np.where(counts > min_size)[0]
    return np.isin(labels, keep_labels)


def compute_dice(prediction, ground_truth, empty_value=1.0):
    """Voxel-wise Dice = 2 * |intersection| / (|prediction| + |ground truth|).
    If both masks are empty the score is `empty_value` (1.0); if only one is empty
    the score is 0."""
    prediction = np.asarray(prediction).astype(bool)
    ground_truth = np.asarray(ground_truth).astype(bool)
    total = prediction.sum() + ground_truth.sum()
    if total == 0:
        return empty_value
    return 2.0 * np.logical_and(prediction, ground_truth).sum() / total


def _surface_distances(a, b, spacing):
    """Distances from the surface voxels of `a` to the nearest surface of `b` and the
    reverse, in millimetres."""
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    a_surface = a & ~binary_erosion(a)
    b_surface = b & ~binary_erosion(b)
    dist_to_b = distance_transform_edt(~b_surface, sampling=spacing)
    dist_to_a = distance_transform_edt(~a_surface, sampling=spacing)
    return dist_to_b[a_surface], dist_to_a[b_surface]


def hausdorff95(prediction, ground_truth, spacing):
    """95th-percentile (symmetric) Hausdorff distance in millimetres."""
    a_to_b, b_to_a = _surface_distances(prediction, ground_truth, spacing)
    if a_to_b.size == 0 or b_to_a.size == 0:
        return None
    return float(max(np.percentile(a_to_b, 95), np.percentile(b_to_a, 95)))


def assd(prediction, ground_truth, spacing):
    """Average symmetric surface distance in millimetres."""
    a_to_b, b_to_a = _surface_distances(prediction, ground_truth, spacing)
    if a_to_b.size == 0 or b_to_a.size == 0:
        return None
    return float((a_to_b.sum() + b_to_a.sum()) / (a_to_b.size + b_to_a.size))


def get_aggregates(records):
    """Aggregate Dice in three strata (overall, lesion-only, no-lesion) and the
    surface metrics over the cases where they are defined, then write metrics.json."""
    def mean_or_none(values):
        clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(clean)) if clean else None

    overall_dice = [r["Dice"] for r in records]
    lesion_dice = [r["Dice"] for r in records if r["ground_truth_has_lesion"] == 1]
    no_lesion_dice = [r["Dice"] for r in records if r["ground_truth_has_lesion"] == 0]
    hd95_values = [r["HD95"] for r in records]
    assd_values = [r["ASSD"] for r in records]

    aggregates = {
        "overall_Dice_mean": mean_or_none(overall_dice),
        "lesion_Dice_mean": mean_or_none(lesion_dice),
        "NO_lesion_Dice_mean": mean_or_none(no_lesion_dice),
        "HD95_mean": mean_or_none(hd95_values),
        "ASSD_mean": mean_or_none(assd_values),
        "n_cases": len(records),
        "n_lesion_cases": len(lesion_dice),
        "n_no_lesion_cases": len(no_lesion_dice),
        "n_surface_metric_cases": len([v for v in hd95_values if v is not None]),
    }

    pd.DataFrame(
        [{"Metric": k, "Value": v} for k, v in aggregates.items()]
    ).to_csv(OUTPUT_DIRECTORY / "average_metrics.csv", index=False)

    metrics = {"results": records, "aggregates": aggregates}
    with open(OUTPUT_DIRECTORY / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("\t | Metrics written: "
          f"overall_Dice={aggregates['overall_Dice_mean']}, "
          f"lesion_Dice={aggregates['lesion_Dice_mean']}, "
          f"NO_lesion_Dice={aggregates['NO_lesion_Dice_mean']}, "
          f"HD95={aggregates['HD95_mean']}, ASSD={aggregates['ASSD_mean']}")


if __name__ == "__main__":
    raise SystemExit(main())
