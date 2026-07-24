"""
inference.py
------------
Core prediction pipeline:
  1. Out-of-distribution (OOD) check via autoencoder reconstruction error.
  2. Crop disease classification via the CNN model.

OOD_THRESHOLD matches the value used during model development/testing.
Tune this if you recalibrate the autoencoder on new data.
"""

import torch.nn.functional as F

OOD_THRESHOLD = 0.015


def autoencoder_reconstruction_loss(autoencoder_model, img_tensor, device):
    """Mean-squared reconstruction error between input and autoencoder output."""
    img_tensor = img_tensor.to(device)
    autoencoder_model.eval()

    import torch

    with torch.no_grad():
        reconstruction = autoencoder_model(img_tensor)

    error = F.mse_loss(reconstruction[0], img_tensor, reduction="none")
    loss = error.mean(dim=(1, 2, 3)).cpu().numpy()
    loss = float(loss[0])
    return loss


def ood_decision(reconstruction_loss, threshold=OOD_THRESHOLD):
    """
    Returns "proceed" if the image looks in-distribution (low reconstruction
    error), otherwise "not_proceed".
    """
    if reconstruction_loss >= threshold:
        return "not_proceed"
    return "proceed"


def run_ood_check(image, autoencoder_model, device, image_transform_fn):
    """
    Runs the full OOD pipeline on a PIL image.
    Returns (reconstruction_loss, decision).
    """
    img_tensor = image_transform_fn(image)
    reconstruction_loss = autoencoder_reconstruction_loss(autoencoder_model, img_tensor, device)
    decision = ood_decision(reconstruction_loss)
    return reconstruction_loss, decision


def cnn_prediction(cnn_model, img_tensor, device):
    """Runs the CNN classifier and returns (predicted_class_index, confidence)."""
    import torch

    img_tensor = img_tensor.to(device)
    cnn_model.eval()

    with torch.no_grad():
        cnn_output = cnn_model(img_tensor)

    probs = F.softmax(cnn_output, dim=1)
    pred_class = probs.argmax(dim=1)
    pred_prob = probs.max(dim=1).values

    predicted_disease = pred_class.item()
    predicted_prob = pred_prob.item()

    return predicted_disease, round(predicted_prob, 4)


def run_prediction(image, cnn_model, device, cnn_transform_fn):
    """
    Runs the full classification pipeline on a PIL image.
    Returns (predicted_class_index, confidence, cnn_img_tensor).
    The tensor is also returned so it can be reused for Grad-CAM.
    """
    cnn_img_tensor = cnn_transform_fn(image)
    predicted_disease, confidence = cnn_prediction(cnn_model, cnn_img_tensor, device)
    return predicted_disease, confidence, cnn_img_tensor
