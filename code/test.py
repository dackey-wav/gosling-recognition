import torch
from torchvision import transforms, models
from PIL import Image
import sys
import torch.nn as nn
import os

# --- KOD GŁÓWNY ---

if len(sys.argv) < 2:
    print("Użycie: python code/test.py sciezka_do_zdjecia.jpg")
    sys.exit(1)

IMG_PATH = sys.argv[1]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_gosling_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(MODEL_PATH):
    print(f"❌ BŁĄD: Nie znaleziono modelu: {MODEL_PATH}")
    sys.exit(1)

# 1. Model
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model = model.to(DEVICE)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except Exception as e:
    print(f"❌ BŁĄD modelu: {e}")
    sys.exit(1)

model.eval() 

# 2. Obrazek
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    image = Image.open(IMG_PATH).convert('RGB')
except FileNotFoundError:
    print(f"❌ BŁĄD: Nie ma pliku: {IMG_PATH}")
    sys.exit(1)

image_tensor = transform(image).unsqueeze(0).to(DEVICE)

# 3. Predykcja
with torch.no_grad():
    output = model(image_tensor)
    prob_negative = torch.sigmoid(output).item() # Prawdopodobieństwo bycia 'negative'

prob_gosling = 1.0 - prob_negative

print("-" * 30)
# Debug: Pokaż obie wartości
print(f"Prawdopodobieństwo 'Gosling': {prob_gosling:.4f}")
print(f"Prawdopodobieństwo 'Inny':    {prob_negative:.4f}")
print("-" * 30)

# Klasa 0 (Gosling) jest wtedy, gdy prob_negative < 0.5 (czyli prob_gosling > 0.5)
if prob_gosling > 0.5:
    print(f"✅ To JEST Ryan Gosling ({prob_gosling*100:.2f}%)")
else:
    print(f"⛔ To NIE jest Ryan Gosling ({prob_negative*100:.2f}%)")
print("-" * 30)