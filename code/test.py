import torch
from torchvision import transforms
from PIL import Image
import sys
import torch.nn as nn

# Musimy zredefiniować klasę modelu lub zaimportować ją z pliku treningowego
# Dla uproszczenia wklejam definicję klasy (w dużych projektach trzyma się ją w osobny pliku models.py)
class GoslingNet(nn.Module):
    def __init__(self):
        super(GoslingNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.flatten_size = 128 * 28 * 28 
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# --- KOD GŁÓWNY ---

if len(sys.argv) < 2:
    print("Użycie: python predict_pytorch.py sciezka_do_zdjecia.jpg")
    sys.exit(1)

IMG_PATH = sys.argv[1]
MODEL_PATH = "gosling_net.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Przygotowanie modelu
model = GoslingNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval() # Tryb ewaluacji (wyłącza dropout)

# 2. Przygotowanie zdjęcia
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

image = Image.open(IMG_PATH).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(DEVICE) # Dodajemy wymiar batcha (1, 3, 224, 224)

# 3. Predykcja
with torch.no_grad(): # Wyłączamy obliczanie gradientów (oszczędność pamięci)
    output = model(image_tensor)
    probability = torch.sigmoid(output).item() # Zamiana logitu na prawdopodobieństwo 0-1

print(f"Wynik surowy (prawdopodobieństwo): {probability:.4f}")

if probability > 0.5:
    print(f"To JEST Ryan Gosling ({probability*100:.2f}%)")
else:
    print(f"To NIE jest Ryan Gosling ({(1-probability)*100:.2f}%)")