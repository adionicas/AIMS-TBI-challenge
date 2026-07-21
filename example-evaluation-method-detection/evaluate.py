"""
AIMS-TBI 2026 detection evaluation method.

Image-level lesion detection only: for each algorithm job, compare the predicted
lesion presence (a boolean the algorithm writes to the `brain-lesion-presence`
output socket) against the true lesion presence derived from the ground-truth mask.
Report sensitivity, specificity, and balanced accuracy.

Runtime contract (Grand Challenge, 2026 development kit):
  - predictions described in /input/predictions.json (socket schema)
  - prediction file at /input/<job-pk>/output/<output-socket-relative-path>
  - ground truth uploaded separately as a tarball, extracted to
    /opt/ml/input/data/ground_truth/  (NOT bundled in the image)
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import cc3d
import SimpleITK as sitk

# Grand Challenge runtime directories
INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
# Ground truth is provided at runtime by the separately uploaded tarball.
GROUND_TRUTH_DIRECTORY = Path("/opt/ml/input/data/ground_truth")

# Sockets for the 2026 detection phase
INPUT_IMAGE_SLUG = "t1-brain-mri"
OUTPUT_PRESENCE_SLUG = "brain-lesion-presence"

# Manuscript, Data Preparation: "lesions smaller than 10 voxels were excluded".
# A connected component is a real lesion only if it has at least this many voxels.
MIN_LESION_SIZE_VOXELS = 10
CONNECTIVITY = 26


def main():
    predictions = read_predictions()

    # Print what is actually inside the ground-truth directory at runtime, so the
    # filenames and formats of the uploaded tarball (which we may not know ahead of
    # time) are visible in the evaluation logs.
    log_directory(GROUND_TRUTH_DIRECTORY, label="Ground truth")

    records = []
    for job in predictions:
        image_name = get_image_name(values=job["inputs"], slug=INPUT_IMAGE_SLUG)

        # Predicted lesion presence: a JSON boolean on the brain-lesion-presence socket.
        try:
            predicted_presence = bool(read_output_value(job, slug=OUTPUT_PRESENCE_SLUG))
        except Exception as e:
            warnings.warn(f"Could not read prediction for {image_name}: {e}. Skipping.")
            continue

        # True lesion presence, derived from the ground-truth mask with the 10-voxel rule.
        try:
            ground_truth_path = find_ground_truth_path(image_name)
            ground_truth_presence = mask_has_lesion(ground_truth_path)
        except Exception as e:
            warnings.warn(f"Could not read ground truth for {image_name}: {e}. Skipping.")
            continue

        image_type = classify_image_type(ground_truth_presence, predicted_presence)
        print(f"[INFO] {image_name}: ground_truth_lesion={ground_truth_presence}, "
              f"predicted_lesion={predicted_presence} -> {image_type}")

        records.append({
            "image_name": image_name,
            "image_type": image_type,
            "ground_truth_has_lesion": int(ground_truth_presence),
            "prediction_has_lesion": int(predicted_presence),
        })

    pd.DataFrame(records).to_csv(OUTPUT_DIRECTORY / "individual_metrics.csv", index=False)
    print(f"\n \t | Individual metrics written: {OUTPUT_DIRECTORY / 'individual_metrics.csv'}")

    get_aggregates(records)
    return 0


def read_predictions():
    with open(INPUT_DIRECTORY / "predictions.json") as f:
        return json.load(f)


def log_directory(directory, *, label):
    """Print the contents of a directory so the actual files (in particular the
    uploaded ground-truth tarball, whose names and formats we may not know ahead
    of time) are visible in the evaluation logs."""
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
    """Original uploaded image name for the given input socket (used to match ground truth)."""
    for value in values:
        if value["socket"]["slug"] == slug:
            return value["image"]["name"]
    raise RuntimeError(f"Input image for socket '{slug}' not found")


def get_socket_relative_path(*, values, slug):
    for value in values:
        if value["socket"]["slug"] == slug:
            return value["socket"]["relative_path"]
    raise RuntimeError(f"Socket '{slug}' not found")


def read_output_value(job, *, slug):
    """Read the algorithm's JSON output for an output socket. Reads the file the
    algorithm wrote; falls back to the value inlined in predictions.json."""
    relative_path = get_socket_relative_path(values=job["outputs"], slug=slug)
    location = INPUT_DIRECTORY / job["pk"] / "output" / relative_path
    if location.exists():
        with open(location) as f:
            return json.load(f)
    for value in job["outputs"]:
        if value["socket"]["slug"] == slug:
            return value["value"]
    raise FileNotFoundError(f"Prediction for socket '{slug}' not found at {location}")


def extract_subject_id(image_name):
    """Subject id from the input image name, e.g. 'scan_0632_T1.nii.gz' -> 'scan_0632'.
    Handles both '_T1' and '_T1w' style modality markers and any image extension."""
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
    ground-truth directory recursively. Accepts either .mha or NIfTI, and either
    'lesion' or 'Lesion' (with or without a '_bin' suffix), so the container does not
    depend on the exact mask name or format."""
    subject = extract_subject_id(image_name)
    subject_lower = subject.lower()

    candidates = []
    for root, _dirs, files in os.walk(GROUND_TRUTH_DIRECTORY):
        for filename in files:
            low = filename.lower()
            if not (low.endswith(".mha") or low.endswith(".nii.gz") or low.endswith(".nii")):
                continue
            # Delimited match so e.g. scan_0003 does not match scan_00031
            if (subject_lower + "_") in low or (subject_lower + ".") in low:
                candidates.append(os.path.join(root, filename))

    if not candidates:
        raise FileNotFoundError(
            f"No ground-truth mask for subject '{subject}' under {GROUND_TRUTH_DIRECTORY}"
        )

    def preference(path):
        low = os.path.basename(path).lower()
        return (
            0 if "lesion" in low else 1,       # prefer an explicit lesion mask
            0 if low.endswith(".mha") else 1,  # then .mha over NIfTI
            len(low),                          # then the simplest name
        )

    return sorted(candidates, key=preference)[0]


def mask_has_lesion(mask_path, min_size=MIN_LESION_SIZE_VOXELS, connectivity=CONNECTIVITY):
    """Image-level lesion presence from a ground-truth mask. True if the mask contains
    at least one connected component of `min_size` voxels or more. Components smaller
    than `min_size` voxels are excluded (manuscript: lesions under 10 voxels excluded)."""
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return False
    labels, n = cc3d.connected_components(mask, connectivity=connectivity, return_N=True)
    if n == 0:
        return False
    counts = np.bincount(labels.reshape(-1))
    counts[0] = 0  # ignore the background label
    return bool((counts >= min_size).any())


def classify_image_type(gt_has_lesion, pred_has_lesion):
    """Image-level confusion-matrix label."""
    if gt_has_lesion and pred_has_lesion:
        return "True Positive"
    if (not gt_has_lesion) and (not pred_has_lesion):
        return "True Negative"
    if (not gt_has_lesion) and pred_has_lesion:
        return "False Positive"
    return "False Negative"  # ground truth has a lesion, prediction says none


def get_aggregates(records):
    """Sensitivity, specificity, and balanced accuracy from the image-level labels."""
    tp = sum(1 for r in records if r["image_type"] == "True Positive")
    tn = sum(1 for r in records if r["image_type"] == "True Negative")
    fp = sum(1 for r in records if r["image_type"] == "False Positive")
    fn = sum(1 for r in records if r["image_type"] == "False Negative")

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2.0

    aggregates = {
        "Total_True_Positives": int(tp),
        "Total_True_Negatives": int(tn),
        "Total_False_Positives": int(fp),
        "Total_False_Negatives": int(fn),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Balanced_Accuracy": float(balanced_accuracy),
    }

    pd.DataFrame(
        [{"Metric": k, "Value": v} for k, v in aggregates.items()]
    ).to_csv(OUTPUT_DIRECTORY / "average_metrics.csv", index=False)

    metrics = {"results": records, "aggregates": aggregates}
    with open(OUTPUT_DIRECTORY / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\t | Metrics written: Sensitivity={sensitivity:.4f}, "
          f"Specificity={specificity:.4f}, Balanced_Accuracy={balanced_accuracy:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
