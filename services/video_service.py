import os
import uuid
import cv2
import numpy as np
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from config import VIDEO_MODEL_PATH, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, TEMP_FOLDER, VIDEO_FRAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

video_model = None


def load_video_model():
    global video_model
    if video_model is None:
        model = timm.create_model("efficientvit_b0", pretrained=False, num_classes=1, drop_rate=0.3)
        model.load_state_dict(torch.load(VIDEO_MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        video_model = model
    return video_model


def get_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def extract_frames(video_path, n_frames=VIDEO_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames


def predict_video(file_path):
    model = load_video_model()
    transform = get_transform()
    frames = extract_frames(file_path)

    all_probs = []
    frame_data = []

    for frame_rgb in frames:
        tensor = transform(image=frame_rgb)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = model(tensor)
        prob = torch.sigmoid(logit).item()
        all_probs.append(prob)
        frame_data.append((frame_rgb, prob))

    avg_prob = np.mean(all_probs)
    label = "Real" if avg_prob >= 0.5 else "Fake"
    confidence = avg_prob * 100 if avg_prob >= 0.5 else (1 - avg_prob) * 100

    frame_data.sort(key=lambda x: abs(x[1] - 0.5), reverse=True)
    top_frames = frame_data[:5]

    gradcam_path = generate_video_gradcam(model, top_frames)

    # Use the most representative frame (highest confidence) for RGB histogram
    representative_frame = top_frames[0][0]
    rgb_path = generate_rgb_histogram(representative_frame)

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "gradcam_path": gradcam_path,
        "rgb_path": rgb_path,
        "file_type": "video"
    }


def generate_video_gradcam(model, top_frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_layers = [model.stages[-1]]
    transform = get_transform()

    os.makedirs(TEMP_FOLDER, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    cam = GradCAM(model=model, target_layers=target_layers)

    for i, (frame_rgb, prob) in enumerate(top_frames):
        tensor = transform(image=frame_rgb)["image"].unsqueeze(0).to(DEVICE)
        image_float = frame_rgb.astype(np.float32) / 255.0

        targets = [BinaryClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
        visualization = show_cam_on_image(image_float, grayscale_cam, use_rgb=True)

        label = "Real" if prob >= 0.5 else "Fake"
        conf = prob * 100 if prob >= 0.5 else (1 - prob) * 100

        axes[0][i].imshow(frame_rgb)
        axes[0][i].set_title(f"Frame {i+1}", fontsize=9, fontweight="bold")
        axes[0][i].axis("off")

        axes[1][i].imshow(visualization)
        axes[1][i].set_title(f"{label} ({conf:.1f}%)", fontsize=9, fontweight="bold")
        axes[1][i].axis("off")

    plt.suptitle("Top 5 Confident Frames — GradCAM Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()

    gradcam_filename = f"gradcam_video_{uuid.uuid4().hex}.jpg"
    gradcam_full_path = os.path.join(TEMP_FOLDER, gradcam_filename)
    plt.savefig(gradcam_full_path, dpi=100, bbox_inches="tight")
    plt.close()

    return f"temp/{gradcam_filename}"


def generate_rgb_histogram(frame_rgb):
    """Generate RGB channel frequency distribution histogram for a video frame."""
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
        channel = frame_rgb[:, :, idx].flatten()
        hist, bins = np.histogram(channel, bins=256, range=(0, 256))
        ax.plot(bins[:-1], hist, color=color, alpha=0.85, linewidth=1.8, label=name)
        ax.fill_between(bins[:-1], hist, alpha=0.12, color=color)

    ax.set_xlabel("Pixel Intensity (0\u2013255)", fontsize=10, fontweight="bold", color="#374151")
    ax.set_ylabel("Frequency", fontsize=10, fontweight="bold", color="#374151")
    ax.set_title("RGB Channel Frequency Distribution (Most Confident Frame)", fontsize=11, fontweight="bold", color="#111827", pad=10)
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