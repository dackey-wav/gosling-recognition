import os
import random
import shutil

# === PROJECT PATHS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # data-scripts
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))   # project root

SRC = os.path.join(PROJECT_ROOT, "faces")
DST = os.path.join(PROJECT_ROOT, "dataset")

CLASSES = ["gosling", "negative"]
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}

random.seed(42)  # reproducible split

print(f"[INFO] Source: {SRC}")
print(f"[INFO] Dest:   {DST}")

# === CREATE DIRECTORIES ===
for split in SPLITS:
    for cls in CLASSES:
        os.makedirs(os.path.join(DST, split, cls), exist_ok=True)

# === SPLIT ===
for cls in CLASSES:
    class_dir = os.path.join(SRC, cls)

    if not os.path.exists(class_dir):
        print(f"[WARNING] Missing folder: {class_dir}")
        continue

    files = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    files.sort()           # stable order
    random.shuffle(files)  # randomize

    n = len(files)
    train_end = int(SPLITS["train"] * n)
    val_end = train_end + int(SPLITS["val"] * n)

    split_map = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    print(f"\n[INFO] Class '{cls}' → total: {n}")
    print(f" train: {len(split_map['train'])}")
    print(f" val:   {len(split_map['val'])}")
    print(f" test:  {len(split_map['test'])}")

    for split, fls in split_map.items():
        for f in fls:
            src_path = os.path.join(class_dir, f)
            dst_path = os.path.join(DST, split, cls, f)

            # Skip if already exists
            if os.path.exists(dst_path):
                continue

            try:
                shutil.copy2(src_path, dst_path)
            except Exception:
                continue

print("\n[INFO] Dataset split DONE ✅")
