import os
import time
from config import ALLOWED_IMAGE_EXTENSIONS, ALLOWED_VIDEO_EXTENSIONS, TEMP_FOLDER


def allowed_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def allowed_video(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def allowed_file(filename):
    return allowed_image(filename) or allowed_video(filename)


def is_video(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def is_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def get_temp_path(filename):
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    timestamp = str(int(time.time()))
    name, ext = os.path.splitext(filename)
    return os.path.join(TEMP_FOLDER, f"{name}_{timestamp}{ext}")


def cleanup_old_files(max_age_seconds=3600):
    if not os.path.exists(TEMP_FOLDER):
        return
    now = time.time()
    for fname in os.listdir(TEMP_FOLDER):
        fpath = os.path.join(TEMP_FOLDER, fname)
        if os.path.isfile(fpath):
            if now - os.path.getmtime(fpath) > max_age_seconds:
                os.remove(fpath)