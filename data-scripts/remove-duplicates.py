import os
from PIL import Image
import imagehash


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
RAW_IMAGES_DIR = os.path.join(PROJECT_ROOT, "raw_images")
CLASSES = ["gosling", "negative"]



def remove_duplicates(cls_name):
    folder = os.path.join(RAW_IMAGES_DIR, cls_name)

    if not os.path.exists(folder):
        print(f"[WARN] Folder not found: {folder}")
        return

    hashes = {}
    removed_count = 0

    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)

        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        try:
            with Image.open(path) as img:
                h = imagehash.phash(img)

            if h in hashes:
                try:
                    os.remove(path)
                    removed_count += 1
                    print(f"[INFO] Removed duplicate: {path}")
                except Exception as e:
                    print(f"[ERROR] Cannot remove {path}: {e}")
            else:
                hashes[h] = fname

        except Exception:
            try:
                os.remove(path)
                removed_count += 1
                print(f"[INFO] Removed invalid image: {path}")
            except Exception as e:
                print(f"[ERROR] Cannot remove {path}: {e}")

    print(f"[INFO] {cls_name}: removed {removed_count} duplicates/invalid images")


for cls in CLASSES:
    remove_duplicates(cls)

print("[INFO] Duplicate removal done!")
