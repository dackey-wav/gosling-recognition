import os
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
RAW_IMAGES_DIR = os.path.join(PROJECT_ROOT, "raw_images")
CLASSES = ["gosling", "negative"]
MIN_SIZE = 80


def clean_folder(cls_name):
    folder = os.path.join(RAW_IMAGES_DIR, cls_name)

    if not os.path.exists(folder):
        print(f"[WARN] Folder not found: {folder}")
        return

    removed_count = 0
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)

        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        try:
            with Image.open(path) as img:
                img.verify()

            with Image.open(path) as img:
                w, h = img.size
                if min(w, h) < MIN_SIZE:
                    os.remove(path)
                    removed_count += 1

        except Exception:
            try:
                os.remove(path)
                removed_count += 1
            except Exception as e:
                print(f"[ERROR] Cannot remove {path}: {e}")

    print(f"[INFO] {cls_name}: removed {removed_count} invalid/small images")



for cls in CLASSES:
    clean_folder(cls)

print("[INFO] Cleaning done!")
