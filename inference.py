import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import ToPILImage
import halcon as ha

from anomalib.data import MVTec
from anomalib.data.utils import read_image
from anomalib.deploy import OpenVINOInferencer
from anomalib.models import Padim
import os
from pathlib import Path
from typing import Any

from git.repo import Repo

# def show_image_and_mask(sample: dict[str, Any], index: int) -> Image:
#     """Show an image with a mask.

#     Args:
#         sample (dict[str, Any]): Sample from the dataset.
#         index (int): Index of the sample.

#     Returns:
#         Image: Output image with a mask.
#     """
#     # Load the image from the path
#     image = Image.open(sample["image_path"][index])

#     # Load the mask and convert it to RGB
#     mask = ToPILImage()(sample["mask"][index]).convert("RGB")

#     # Resize mask to match image size, if they differ
#     if image.size != mask.size:
#         mask = mask.resize(image.size)

#     return Image.fromarray(np.hstack((np.array(image), np.array(mask))))


# show_image_and_mask(data, index=0)

openvino_model_path = "weights/openvino/model.bin"
metadata =  "weights/openvino/metadata.json"
inferencer = OpenVINOInferencer(
    path=openvino_model_path,  # Path to the OpenVINO IR model.
    metadata=metadata,  # Path to the metadata file.
    device="CPU",  # We would like to run it on an Intel CPU.
)


AcqHandle = ha.open_framegrabber ('USB3Vision', 0, 0, 0, 0, 0, 0, 'progressive', -1, 'default', -1, 'false', 'default', "26760151C87C_Basler_acA192040uc", 0, -1)
ha.set_framegrabber_param (AcqHandle, 'AcquisitionFrameRate', 10)
ha.set_framegrabber_param (AcqHandle, 'ExposureTime', 1500)
ha.set_framegrabber_param (AcqHandle, 'Gain', 5)

plt.ion()
fig, ax = plt.subplots(figsize=(20, 15))

while True:
    ha.grab_image_async (AcqHandle, -1)
    image = ha.grab_image(AcqHandle)
    np_img = ha.himage_as_numpy_array(image)
    predictions = inferencer.predict(image=np_img)
    
    # Clear the previous image and plot the new one
    ax.clear()
    ax.imshow(predictions.heat_map)

    plt.pause(0.01)