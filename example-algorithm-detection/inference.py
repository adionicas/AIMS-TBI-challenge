"""
Example DETECTION algorithm for the AIMS-TBI 2026 challenge.

Detection is image-level: for each T1 scan decide whether a lesion is present and
write a single boolean to the `brain-lesion-presence` socket. This is NOT a mask.

Workflow:
  ./test_run.sh   build the container and run it on the bundled test input
  ./save.sh       produce the .tar.gz image to upload to Grand Challenge

Replace predict_presence() with your own detector. Everything else can stay.
"""

import json
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")


def run():
    # Read the input T1 (provided as .mha at runtime)
    t1_brain_mri = load_image_file_as_array(location=INPUT_PATH / "images/t1-brain-mri")

    # Decide presence
    lesion_present = predict_presence(t1_brain_mri)

    # Write the single boolean to the brain-lesion-presence socket
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH / "brain-lesion-presence.json", "w") as f:
        json.dump(bool(lesion_present), f)

    print(f"Wrote brain-lesion-presence.json = {bool(lesion_present)}")
    return 0


def load_image_file_as_array(*, location):
    input_files = glob(str(location / "*.mha"))
    image = SimpleITK.ReadImage(input_files[0])
    return SimpleITK.GetArrayFromImage(image)


def predict_presence(image):
    # Replace this with your real detector.
    # Return True if a lesion is present, False otherwise.
    # This placeholder always returns True so the example runs end to end.
    _ = np.asarray(image)
    return True


if __name__ == "__main__":
    raise SystemExit(run())
