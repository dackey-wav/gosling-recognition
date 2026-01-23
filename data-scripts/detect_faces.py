from facenet_pytorch import MTCNN
import os
from PIL import Image
import torch
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
INPUT_DIR = os.path.join(PROJECT_ROOT, "raw_images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "faces")

CLASSES = ["gosling", "negative"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=False,
    device=device
)

MAX_SIZE = 1200

for cls in CLASSES:
    in_dir = os.path.join(INPUT_DIR, cls)
    out_dir = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(out_dir, exist_ok=True)

    for fname in tqdm(os.listdir(in_dir), desc=f"Processing {cls}"):
        path = os.path.join(in_dir, fname)

        try:
            img = Image.open(path).convert("RGB")
        except:
            continue

        w, h = img.size
        max_side = max(w, h)
        if max_side > MAX_SIZE:
            scale = MAX_SIZE / max_side
            img = img.resize((int(w * scale), int(h * scale)))

        try:
            face = mtcnn(img)
        except Exception:
            continue

        if face is None:
            continue

        # 🔥 Денормализация
        face = face.permute(1, 2, 0)
        face = (face + 1) / 2
        face = (face * 255).clamp(0, 255)
        face = face.byte().cpu().numpy()

        Image.fromarray(face).save(os.path.join(out_dir, fname))

print("[INFO] Face extraction done!")
