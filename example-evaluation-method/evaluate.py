import argparse
import numpy as np
import nibabel as nib
import pandas as pd
import cc3d
import json
import os
import glob
from pathlib import Path
import SimpleITK

INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
GROUND_TRUTH_DIRECTORY = Path("ground_truth")


def main():

    predictions_data = read_predictions()

    # Define metrics list
    metrics = [
        ("Dice_coefficient", compute_dice),
        ("Absolute_volume_difference", compute_absolute_volume_difference),
        ("Absolute_lesion_difference", compute_absolute_lesion_difference),
        ("Lesion-wise_F1-score", compute_lesion_f1_score)
    ]


    # Compute metrics
    metric_values = compute_metrics_3d(predictions_data, metrics)

    metric_values_df = pd.DataFrame(metric_values)
    # metric_values_df.insert(0, 'Segmentation', input_data_files)
    # metric_values_df.insert(1, 'Ground Truth', ground_truth_files)
    metric_values_df.to_csv(OUTPUT_DIRECTORY / "individual_metrics.csv", index=False)

    # Calculate aggregates and export
    get_aggregates(metric_values, metrics)


def read_predictions():
    prediction_file = Path(INPUT_DIRECTORY) / "predictions.json"
    if prediction_file.exists():
        print("Predictions file found")
        with prediction_file.open() as f:
            return json.loads(f.read())
    else:
        print("Predictions file not found")

def compute_metrics_3d(predictions_data, metrics):

    metric_values = {metric_name: [] for metric_name, _ in metrics}
 
    for item in predictions_data:
        segm_pk = item['pk']
        image_name = item['inputs'][0]['image']['name']
        image_pk = item['outputs'][0]['image']['pk']
        ground_truth = load_image_as_array(os.path.join(GROUND_TRUTH_DIRECTORY, image_name.split("_T1.nii.gz")[0] + "_Lesion.mha"))
        tbi_segmentation = load_image_as_array(os.path.join(INPUT_DIRECTORY, segm_pk, "output", "images", "tbi-segmentation", image_pk + ".mha"))
        # Calculate voxel volume
        # voxel_volume = np.prod(nib.load(INPUT_DIRECTORY / input_file).header.get_zooms()[:3]) / 1000
        voxel_volume = 0.001
        # print(f"\n ▁ ▂ ▃ ▅ ▆ ▇ █ Calculating eval metrics █ ▇ ▆ ▅ ▃ ▂ ▁ \n")

        # Calculate metrics for the current pair of files
        for metric_name, metric_function in metrics:
            value = metric_function(tbi_segmentation, ground_truth, voxel_volume) if metric_name == "Absolute_volume_difference" else metric_function(tbi_segmentation, ground_truth)
            # print(f"\t | {metric_name} for image {input_file}: {value}")
            metric_values[metric_name].append(value)

    return metric_values

def compute_dice(im1, im2, empty_value=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size as im1. If not boolean, it will be converted.
    empty_value : scalar, float.

    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        If both images are empty (sum equal to zero) = empty_value

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.

    This function has been adapted from the Verse Challenge repository:
    https://github.com/anjany/verse/blob/main/utils/eval_utilities.py
    """

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_value

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2.0 * intersection.sum() / im_sum

def compute_absolute_volume_difference(im1, im2, voxel_size):
    """
    Computes the absolute volume difference between two masks.

    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    voxel_size : scalar, float (ml)
        If not float, it will be converted.

    Returns
    -------
    abs_vol_diff : float, measured in ml.
        Absolute volume difference as a float.
        Maximum similarity = 0
        No similarity = inf


    Notes
    -----
    The order of inputs is irrelevant. The result will be identical if `im1` and `im2` are switched.
    """

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    # voxel_size = voxel_size.astype(float)
    if not isinstance(voxel_size, float):
        voxel_size = float(voxel_size)

    if im1.shape != im2.shape:
        warnings.warn(
            "Shape mismatch: ground_truth and prediction have difference shapes."
            " The absolute volume difference is computed with mismatching shape masks"
        )

    ground_truth_volume = np.sum(im1) * voxel_size
    prediction_volume = np.sum(im2) * voxel_size
    abs_vol_diff = np.abs(ground_truth_volume - prediction_volume)

    return abs_vol_diff

def compute_absolute_lesion_difference(ground_truth, prediction, connectivity=26):
    """
    Computes the absolute lesion difference between two masks. The number of lesions are counted for
    each volume, and their absolute difference is computed.

    Parameters
    ----------
    ground_truth : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    prediction : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.

    Returns
    -------
    abs_les_diff : int
        Absolute lesion difference as integer.
        Maximum similarity = 0
        No similarity = inf


    Notes
    -----
    """
    ground_truth = np.asarray(ground_truth).astype(bool)
    prediction = np.asarray(prediction).astype(bool)

    _, ground_truth_numb_lesion = cc3d.connected_components(ground_truth, connectivity=connectivity, return_N=True)
    _, prediction_numb_lesion = cc3d.connected_components(prediction, connectivity=connectivity, return_N=True)
    abs_les_diff = abs(ground_truth_numb_lesion - prediction_numb_lesion)

    return abs_les_diff

def compute_lesion_f1_score(ground_truth, prediction, empty_value=1.0, connectivity=26):
    """
    Computes the lesion-wise F1-score between two masks.

    Parameters
    ----------
    ground_truth : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    prediction : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    empty_value : scalar, float.
    connectivity : scalar, int.

    Returns
    -------
    f1_score : float
        Lesion-wise F1-score as float.
        Max score = 1
        Min score = 0
        If both images are empty (tp + fp + fn =0) = empty_value

    Notes
    -----
    This function computes lesion-wise score by defining true positive lesions (tp), false positive lesions (fp) and
    false negative lesions (fn) using 3D connected-component-analysis.

    tp: 3D connected-component from the ground-truth image that overlaps at least on one voxel with the prediction image.
    fp: 3D connected-component from the prediction image that has no voxel overlapping with the ground-truth image.
    fn: 3d connected-component from the ground-truth image that has no voxel overlapping with the prediction image.
    """
    ground_truth = np.asarray(ground_truth).astype(bool)
    prediction = np.asarray(prediction).astype(bool)
    tp = 0
    fp = 0
    fn = 0

    # Check if ground-truth connected-components are detected or missed (tp and fn respectively).
    intersection = np.logical_and(ground_truth, prediction)
    labeled_ground_truth, N = cc3d.connected_components(
        ground_truth, connectivity=connectivity, return_N=True
    )

    # Iterate over ground_truth clusters to find tp and fn.
    # tp and fn are only computed if the ground-truth is not empty.
    if N > 0:
        for _, binary_cluster_image in cc3d.each(labeled_ground_truth, binary=True, in_place=True):
            if np.logical_and(binary_cluster_image, intersection).any():
                tp += 1
            else:
                fn += 1

    # iterate over prediction clusters to find fp.
    # fp are only computed if the prediction image is not empty.
    labeled_prediction, N = cc3d.connected_components(
        prediction, connectivity=connectivity, return_N=True
    )
    if N > 0:
        for _, binary_cluster_image in cc3d.each(labeled_prediction, binary=True, in_place=True):
            if not np.logical_and(binary_cluster_image, ground_truth).any():
                fp += 1

    # Define case when both images are empty.
    if tp + fp + fn == 0:
        _, N = cc3d.connected_components(ground_truth, connectivity=connectivity, return_N=True)
        if N == 0:
            f1_score = empty_value
    else:
        f1_score = tp / (tp + (fp + fn) / 2)

    return f1_score

def get_aggregates(metric_values, metrics):
    """
    Aggregate metrics, output to CSV, and JSON.

    Args:
        metric_values (dict): Dictionary containing metric values for each metric.
        metrics (list): List of tuples containing metric names and corresponding functions.

    Returns:
        dict: Dictionary containing aggregate metrics.
    """
    # Aggregate metrics
    # print(f"\n ▁ ▂ ▃ ▅ ▆ ▇ █ Aggregating eval metrics █ ▇ ▆ ▅ ▃ ▂ ▁ \n")
    aggregate_metrics = {metric_name: (np.mean(values), np.std(values)) for metric_name, values in metric_values.items()}
    # for metric_name, (avg, std) in aggregate_metrics.items():
        # print(f"\t | {metric_name}: {avg} (std: {std})")

    # Output metrics to CSV
    average_metrics_df = pd.DataFrame({
        "Metric": [f"{metric_name}" for metric_name, _ in metrics],
        "Mean Value": [avg for metric_name, (avg, _) in aggregate_metrics.items()],
        "Standard Deviation": [std for _, (_, std) in aggregate_metrics.items()]
    })
    average_metrics_df.to_csv(OUTPUT_DIRECTORY / "average_metrics.csv", index=False)
    print(f" \n \t | Aggregate metrics written: {OUTPUT_DIRECTORY / 'average_metrics.csv'}")

    # Output metrics to JSON
    metrics_data = {
        "results": [],
        "aggregate": {f"{metric_name}_mean": avg for metric_name, (avg, _) in aggregate_metrics.items()},
        "deviation": {f"{metric_name}_sd": std for metric_name, (_, std) in aggregate_metrics.items()}
    }

    for index, values in enumerate(zip(*[metric_values[metric_name] for metric_name, _ in metrics]), start=1):
        metrics_data["results"].append({
            "Volume": index,
            **{metric_name: value for metric_name, value in zip([metric_name for metric_name, _ in metrics], values)}
        })

    # Export to JSON
    write_metrics(metrics=metrics_data)
    print(f"\t | Metrics JSON file written: {OUTPUT_DIRECTORY / 'metrics.json'}")

    return aggregate_metrics

def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    with open(OUTPUT_DIRECTORY / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

def load_image_as_array(location):
    """
    Reads an image using SimpleITK and converts it to a NumPy array.

    Args:
    location (Path): Path to the image file.

    Returns:
    np.ndarray: The image data as a NumPy array.
    """
    # Read the image using SimpleITK
    image = SimpleITK.ReadImage(str(location))

    # Convert the SimpleITK image to a NumPy array
    return SimpleITK.GetArrayFromImage(image)

if __name__ == "__main__":
    main()
