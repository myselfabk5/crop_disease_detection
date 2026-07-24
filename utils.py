"""
utils.py
--------
Image reading & tensor transform helpers shared across the pipeline.
"""

from io import BytesIO

from PIL import Image
from torchvision import transforms


def read_crop_image(image_source):
    """
    Reads an image and returns a PIL RGB image.

    image_source can be:
      - a file path (str / Path)
      - a file-like object (e.g. Streamlit's UploadedFile / camera_input buffer)
      - raw bytes
    """
    if isinstance(image_source, (bytes, bytearray)):
        image_source = BytesIO(image_source)

    img = Image.open(image_source).convert("RGB")
    return img


def image_transform(image):
    """Autoencoder input transform: resize + [0,1] tensor (no normalization)."""
    tensor_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # [0, 1]
        ]
    )
    img_tensor = tensor_transform(image).unsqueeze(0)
    return img_tensor


def cnn_transform(rgb_image):
    """CNN classifier input transform: resize + ImageNet normalization."""
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    cnn_tensor = transform(rgb_image).unsqueeze(0)
    return cnn_tensor
