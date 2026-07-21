import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables
load_dotenv(os.path.join(BASE_DIR, ".env"))

SECRET_KEY = os.getenv("SECRET_KEY", "deepfake_detection_secret_2024")

DATABASE_PATH = os.path.join(BASE_DIR, "deepfake.db")

# Parse comma-separated API keys
api_keys_str = os.getenv("GEMINI_API_KEYS", "")
if api_keys_str:
    GEMINI_API_KEYS = [key.strip() for key in api_keys_str.split(",")]
else:
    GEMINI_API_KEYS = [""]

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
try:
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
except ValueError:
    SMTP_PORT = 587
SMTP_EMAIL = os.getenv("SMTP_EMAIL", "@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "efficientvit_b0_final.pth")
VIDEO_MODEL_PATH = os.path.join(BASE_DIR, "models", "final_video_models", "efficientvit_b0_video_final.pth")

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

TEMP_FOLDER = os.path.join(BASE_DIR, "static", "temp")

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VIDEO_FRAMES = 40
