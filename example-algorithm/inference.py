"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Currently, an example segmentation method is run by the function segment_image(), and you could modify your algorithm version there.

Happy programming!
"""
from pathlib import Path

from glob import glob
import SimpleITK
import numpy as np
import torch

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

def main():
    run()

def run():
    # Read the input
    t1_brain_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/t1-brain-mri",
    )
    
    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    with open(RESOURCE_PATH / "some_resource.txt", "r") as f:
        print(f.read())

    # For now, let us set make bogus predictions
    traumatic_brain_injury_segmentation = segment_image(t1_brain_mri)


    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/tbi-segmentation",
        array=traumatic_brain_injury_segmentation,
    )
    
    return 0


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def write_array_as_image_file(*, location, array):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tiff to match the expected output
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)

def segment_image(image):
    # Here, you would perform the segmentation
    # You would replace this with your actual segmentation algorithm

    gm_min = ((np.max(image)) / 50) * 10
    gm_max = ((np.max(image)) / 50) * 15

    brain_mask1 = np.where(image > gm_min, 1, 0)
    brain_mask2 = np.where(image < gm_max, 1, 0)

    segmented_image = brain_mask1 + brain_mask2
    segmented_image = np.where(segmented_image == 2, 1, 0)

    segmented_image = segmented_image.astype(np.int8)

    return segmented_image

if __name__ == "__main__":
    raise SystemExit(run())