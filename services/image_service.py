import os
import uuid
import numpy as np
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from config import IMAGE_MODEL_PATH, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, TEMP_FOLDER

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_model = None


def load_image_model():
    global image_model
    if image_model is None:
        model = timm.create_model("efficientvit_b0", pretrained=False, num_classes=1)
        model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        image_model = model
    return image_model


def get_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def predict_image(file_path):
    model = load_image_model()
    transform = get_transform()

    raw_image = Image.open(file_path).convert("RGB")
    image_np = np.array(raw_image)
    tensor = transform(image=image_np)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(tensor)

    prob = torch.sigmoid(logit).item()
    label = "Real" if prob >= 0.5 else "Fake"
    confidence = prob * 100 if prob >= 0.5 else (1 - prob) * 100

    gradcam_path = generate_gradcam(model, file_path, image_np)
    rgb_path = generate_rgb_histogram(image_np)

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "gradcam_path": gradcam_path,
        "rgb_path": rgb_path,
        "file_type": "image"
    }


def generate_gradcam(model, file_path, image_np):
    transform = get_transform()
    tensor = transform(image=image_np)["image"].unsqueeze(0).to(DEVICE)

    image_resized = np.array(Image.open(file_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
    image_float = image_resized.astype(np.float32) / 255.0

    target_layers = [model.stages[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [BinaryClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    visualization = show_cam_on_image(image_float, grayscale_cam, use_rgb=True)

    os.makedirs(TEMP_FOLDER, exist_ok=True)
    gradcam_filename = f"gradcam_{uuid.uuid4().hex}.jpg"
    gradcam_full_path = os.path.join(TEMP_FOLDER, gradcam_filename)
    Image.fromarray(visualization).save(gradcam_full_path)

    return f"temp/{gradcam_filename}"


def generate_rgb_histogram(image_np):
    """Generate RGB channel frequency distribution histogram and save to temp folder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f1f5f9")

    channel_cfg = [
        (0, "#ef4444", "Red"),
        (1, "#22c55e", "Green"),
        (2, "#3b82f6", "Blue"),
    ]

    for idx, color, name in channel_cfg:
        channel = image_np[:, :, idx].flatten()
        hist, bins = np.histogram(channel, bins=256, range=(0, 256))
        ax.plot(bins[:-1], hist, color=color, alpha=0.85, linewidth=1.8, label=name)
        ax.fill_between(bins[:-1], hist, alpha=0.12, color=color)

    ax.set_xlabel("Pixel Intensity (0\u2013255)", fontsize=10, fontweight="bold", color="#374151")
    ax.set_ylabel("Frequency", fontsize=10, fontweight="bold", color="#374151")
    ax.set_title("RGB Channel Frequency Distribution", fontsize=12, fontweight="bold", color="#111827", pad=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.set_xlim(0, 255)
    ax.tick_params(colors="#6b7280", labelsize=8)
    ax.grid(True, alpha=0.25, linestyle="--", color="#94a3b8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e2e8f0")
    ax.spines["bottom"].set_color("#e2e8f0")

    plt.tight_layout(pad=1.5)

    os.makedirs(TEMP_FOLDER, exist_ok=True)
    rgb_filename = f"rgb_hist_{uuid.uuid4().hex}.jpg"
    rgb_full_path = os.path.join(TEMP_FOLDER, rgb_filename)
    plt.savefig(rgb_full_path, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    return f"temp/{rgb_filename}"