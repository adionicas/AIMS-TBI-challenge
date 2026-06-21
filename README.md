# AIMS-TBI 2026 Challenge

Automated Identification of Moderate-Severe Traumatic Brain Injury Lesions, third edition, MICCAI 2026.
Register and submit at https://aims-tbi26.grand-challenge.org/

This repository provides the runtime contract and example algorithms for both challenge tasks.

The 2026 challenge has two separate phases, each with its own leaderboard:

- **Detection**: for each scan, decide whether a lesion is present. Image-level classification.
- **Segmentation**: delineate the lesion voxels.

Training data are multimodal MRI. Evaluation uses the T1-weighted scan only.

## What your algorithm reads and writes

Every algorithm receives one input and must produce one output, defined by Grand Challenge "sockets". At runtime the input image is always provided as `.mha`.

**Input (both tasks)**
- Socket `t1-brain-mri`. Provided at `/input/images/t1-brain-mri/<id>.mha`.

**Output for Detection**
- Socket `brain-lesion-presence`. Write a single JSON boolean (`true` if a lesion is present, `false` otherwise) to `/output/brain-lesion-presence.json`.

**Output for Segmentation**
- Socket `traumatic-brain-injury-segmentation`. Write a binary mask (0 background, 1 lesion) to `/output/images/tbi-segmentation/output.mha`, on the same grid as the input image.

Detection is a single boolean, not a mask. Segmentation is a mask, not a boolean. Use the example that matches the phase you are entering.

## Examples

- `example-algorithm-detection/` writes the boolean output for the Detection phase.
- `example-algorithm-segmentation/` writes the mask output for the Segmentation phase.

Each folder uses the same workflow:

1. Run `./test_run.sh` to build the container and run it on the bundled test input, writing to `test/output`.
2. Replace the prediction function with your own method.
3. Run `./save.sh` to produce the `.tar.gz` image you upload to Grand Challenge.

## How to submit

1. On the challenge site go to **Submit** and choose the phase (Detection or Segmentation).
2. If you have no algorithm yet, use the link there to create one. It pre-sets the correct input and output sockets for that phase, so do not build a standalone algorithm with sockets chosen by hand.
3. Upload the `.tar.gz` from `save.sh` as the algorithm's container image and wait for it to become active.
4. **GPU**: the Segmentation phase requires the NVIDIA T4 tier. Set it on the algorithm under **Job requires gpu type**. The Detection example uses no GPU, so No GPU is fine there. The GPU type is a setting on the algorithm, not inside the container image, and cannot be changed on the image page.
5. Submit to the phase.

## Evaluation metrics

**Detection** (computed per image, then averaged)
- Sensitivity, Specificity, Balanced accuracy. A scan counts as lesion-positive when its ground truth contains a connected lesion of at least 10 voxels.

**Segmentation** (computed per scan, then averaged)
- Dice (overall, lesion-only, no-lesion), Hausdorff distance 95th percentile, average symmetric surface distance. Connected components of 50 voxels or fewer are removed from both masks before scoring. The two surface distances are reported only where both masks are non-empty.

## Ground truth

Ground truth is held by the organizers and supplied to the evaluation at runtime. It is not part of this repository.

## Local testing tips

- Test locally before uploading. The website environment is slower to iterate on.
- The container runs with no internet (`--network none`). Include any model weights inside the image.

## Questions

- Emily Dennis: Emily.Dennis@hsc.utah.edu
- Adrian Onicas: adrian.onicas@hsc.utah.edu
