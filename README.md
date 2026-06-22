# AIMS-TBI-challenge

This repository provides baseline example algorithms for the [AIMS-TBI 2026 Challenge](https://aims-tbi26.grand-challenge.org/). Participants can use these templates to develop and package their methods as Docker containers for evaluation on the challenge platform.

## Challenge Description

AIMS-TBI 2026 is the 3rd iteration of the Automated Identification of Moderate-Severe
Traumatic Brain Injury Lesions challenge, held at MICCAI 2026 in association with the
BraTS challenges and BrainWorks workshop. This year the challenge features two tasks —
**lesion detection** and **lesion segmentation** — with dual leaderboards.

Visit the [official challenge page](https://aims-tbi26.grand-challenge.org/) for registration and submissions.

For the complete challenge design document, see the
[challenge proposal](307-AIMS-TBI_-_Automated_Identification_of_Moderate-Severe_Traumatic_Brain_2026-02-23T21-16-18%20(1).pdf).

## Example Algorithms

This year there are two tasks, so there are two example folders. Use the one for the task you are entering:

* **`example-algorithm-detection/`** — for lesion detection. It outputs a single true/false answer indicating whether a lesion is present.
* **`example-algorithm-segmentation/`** — for lesion segmentation. It outputs a lesion mask.

## Getting Started

1. **Test Run:** In the example folder for your task, execute `test_run.sh` to run the Python script `inference.py`. This will give you a basic idea of how the process works.

2. **Implement Your Method:** Open `inference.py` and modify the prediction function to include your own algorithm. For detection, return whether a lesion is present; for segmentation, return your lesion mask.

3. **Save Docker Image:** Once you're satisfied with your results, run `save.sh` to create the Docker image. This image is what you'll submit to the challenge.

## Preliminary Development Phase

The preliminary development phase is intended for testing your container, that is, confirming it builds, runs, and produces valid output on the platform. The number of ground-truth images in this phase is small, so the scores it returns are not a reliable estimate of your method's performance. Use it to check that your submission works, not to judge accuracy.

## Submissions

Only **2 submissions** are allowed. Please test your method thoroughly before submitting.

## Recommendations

* **Test Locally:** It's strongly recommended to thoroughly test your implementation locally before generating the final Docker image. The challenge website's environment can be time-consuming.

* **GPU:** The segmentation task runs on an NVIDIA T4 GPU. After uploading your algorithm, set its required GPU type to T4 on the Grand Challenge algorithm page.

## Questions?

If you have any questions or need further assistance, please don't hesitate to contact us:

* Emily Dennis: Emily.Dennis@hsc.utah.edu
* Adrian Onicas: adrian.onicas@hsc.utah.edu
