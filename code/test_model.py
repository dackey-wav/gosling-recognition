import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- KONFIGURACJA ---
TEST_DIR = '../dataset/test' # Upewnij się, że ścieżka jest poprawna
IMG_SIZE = 224
BATCH_SIZE = 32
# Tutaj podajemy NOWY plik z wagami!
MODEL_PATH = "best_gosling_model.pth" 

# Wykrywanie sprzętu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Testowanie na: {device}")

# 1. Transformacje (Muszą być identyczne jak w train_optimized.py!)
# ResNet wymaga normalizacji ImageNet, a nie 0.5/0.5/0.5
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 2. Ładowanie danych
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Odtworzenie architektury ResNet18
print("🧠 Budowanie modelu ResNet18...")
model = models.resnet18(weights=None) # Nie pobieramy wag z neta, bo zaraz załadujemy swoje
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1) # Zmieniamy ostatnią warstwę tak jak w treningu

# Przenosimy model na GPU
model = model.to(device)

# 4. Ładowanie Twoich wytrenowanych wag
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"✅ Załadowano wagi z pliku: {MODEL_PATH}")
else:
    print(f"❌ BŁĄD: Nie znaleziono pliku {MODEL_PATH}!")
    exit()

model.eval()

# 5. Wielki Test
all_preds = []
all_labels = []

print(f"Rozpoczynam testowanie na {len(test_dataset)} obrazkach...")

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        # ResNet nie ma sigmoida na końcu w treningu (BCEWithLogitsLoss),
        # więc musimy go dodać teraz ręcznie, żeby dostać prawdopodobieństwo.
        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5 
        
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.numpy())

# 6. Wyniki i Raport
print("\n--- RAPORT KLASYFIKACJI (ResNet18) ---")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

# Macierz pomyłek
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes, cmap='Blues')
plt.xlabel('Przewidziane')
plt.ylabel('Prawdziwe')
plt.title('Macierz Pomyłek - ResNet18')

# Zapis wykresu do pliku zamiast show()
plt.savefig('wynik_resnet.png')
print("Wykres zapisano jako 'wynik_resnet.png'")