# AIMS-TBI-challenge

Thank you for submitting to the AIMS-TBI segmentation challenge! 

This repository provides an example method for the AIMS TBI Segmentation Challenge.

## Challenge Description

AIMS-TBI 2026 is the 3rd iteration of the Automated Identification of Moderate-Severe
Traumatic Brain Injury Lesions challenge, held at MICCAI 2026 in association with the
BraTS challenges and BrainWorks workshop. This year the challenge features two tasks —
**lesion detection** and **lesion segmentation** — with dual leaderboards and multimodal
MRI training data. Evaluation is performed on T1-weighted MRI only.

For the complete challenge design document, see the
[BIAS challenge proposal](307-AIMS-TBI_-_Automated_Identification_of_Moderate-Severe_Traumatic_Brain_2026-02-23T21-16-18%20(1).pdf).

## Getting Started

1. **Test Run:** Execute `test_run.sh` to run the Python script `inference.py`. This will give you a basic idea of how the process works.

2. **Implement Your Segmentation:** Open `inference.py` and modify the `segment_image` function to include your own segmentation algorithm. **Important:** Please keep the line `"segmented_image = segmented_image.astype(np.int8)"` unchanged. This is necessary for compatibility with the challenge website.

3. **Save Docker Image:** Once you're satisfied with your segmentation results, run `save.sh` to create the Docker image. This image is what you'll submit to the challenge.

## Recommendations

* **Test Locally:** It's strongly recommended to thoroughly test your implementation locally before generating the final Docker image. The challenge website's environment can be time-consuming.

## Questions?

If you have any questions or need further assistance, please don't hesitate to contact us:

* Emily Dennis: Emily.Dennis@hsc.utah.edu
* Adrian Onicas: adrian.onicas@hsc.utah.edu
