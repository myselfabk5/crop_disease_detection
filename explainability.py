"""
explainability.py
------------------
Grad-CAM based visual explanation of the CNN's prediction.

Produces two images:
  1. The (normalized-then-denormalized) original image.
  2. The same image overlaid with a Grad-CAM heatmap showing where the
     model focused to make its prediction.
"""

import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def grad_cam_transform(image):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ]
    )
    return transform(image)


def grad_cam_image(cnn_model, cnn_img_tensor):
    """Computes the Grad-CAM grayscale activation map for the top predicted class."""
    target_layers = [cnn_model.blocks[-1]]

    cam = GradCAM(model=cnn_model, target_layers=target_layers)
    targets = None

    grayscale_cam = cam(input_tensor=cnn_img_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]

    return grayscale_cam


def build_gradcam_images(grayscale_cam, org_image):
    """
    Returns (original_image_pil, gradcam_overlay_pil) suitable for
    st.image() display, instead of the notebook's matplotlib.pyplot.show().
    """
    org_tensor = grad_cam_transform(org_image)
    img_np = org_tensor.permute(1, 2, 0).numpy()

    img_np = img_np * IMAGENET_STD + IMAGENET_MEAN
    img_np = np.clip(img_np, 0, 1)

    grad_cam_visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    original_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    gradcam_pil = Image.fromarray(grad_cam_visualization.astype(np.uint8))

    return original_pil, gradcam_pil


def run_explainability(cnn_model, cnn_img_tensor, original_image, device):
    """
    Full explainability pipeline. Returns (original_pil, gradcam_pil).
    cnn_img_tensor is the (1, 3, 224, 224) normalized tensor already used
    for prediction; it is moved onto `device` to match the model.
    """
    cnn_img_tensor = cnn_img_tensor.to(device)
    grayscale_cam = grad_cam_image(cnn_model, cnn_img_tensor)
    return build_gradcam_images(grayscale_cam, original_image)
