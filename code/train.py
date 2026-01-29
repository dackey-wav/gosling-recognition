import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import copy
from tqdm import tqdm  # Pasek postępu

# --- KONFIGURACJA ---
TRAIN_DIR = '../dataset/train' # Upewnij się, że ścieżka jest poprawna!
VAL_DIR = '../dataset/val'
IMG_SIZE = 224
BATCH_SIZE = 64        # Zwiększamy, bo RTX 5070 to potwór
EPOCHS = 20
LEARNING_RATE = 0.0001 # Mniejszy LR, bo używamy wytrenowanej sieci (fine-tuning)
NUM_WORKERS = 4        # Użycie procesora do ładowania danych w tle

# Wykrywanie sprzętu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Używam urządzenia: {device}")
if device.type == 'cuda':
    print(f"   Karta: {torch.cuda.get_device_name(0)}")

# 1. Zaawansowane Transformacje (Augmentacja)
# Używamy statystyk ImageNet dla normalizacji (wymagane dla ResNet)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),           # Lekki obrót
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Zmiana oświetlenia
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 2. Dataset i Loader (zoptymalizowane pod GPU)
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=NUM_WORKERS, pin_memory=True)

print(f"📂 Klasy: {train_dataset.classes}")
print(f"📊 Dane treningowe: {len(train_dataset)} | Walidacyjne: {len(val_dataset)}")

# 3. Model: Transfer Learning (ResNet18)
print("🧠 Pobieranie modelu ResNet18...")
model = models.resnet18(weights='IMAGENET1K_V1')

# "Zamrażamy" początkowe warstwy (opcjonalnie - na start warto trenować tylko koniec)
# for param in model.parameters():
#     param.requires_grad = False

# Podmieniamy ostatnią warstwę (która oryginalnie ma 1000 klas) na naszą (1 klasa)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1) # Wyjście binarne (logit)

model = model.to(device)

# 4. Optymalizator i Loss
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Scheduler: Zmniejsz learning rate, jeśli strata (loss) przestanie spadać
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

# Mixed Precision Scaler (dla kart RTX - przyspiesza trening)
scaler = torch.amp.GradScaler('cuda')

# 5. Funkcja trenująca
def train_model(model, train_loader, val_loader, epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(epochs):
        print(f'\nEpoka {epoch+1}/{epochs}')
        print('-' * 10)

        # Każda epoka ma fazę treningu i walidacji
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Pasek postępu (tqdm)
            loop = tqdm(dataloader, leave=True)
            
            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                optimizer.zero_grad()

                # Mixed Precision Context
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        preds = torch.sigmoid(outputs) > 0.5

                    # Backprop tylko w fazie train
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                # Statystyki
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Aktualizacja paska postępu
                loop.set_description(f"{phase.upper()}")
                loop.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep Copy modelu jeśli jest najlepszy
            if phase == 'val':
                scheduler.step(epoch_loss) # Aktualizacja schedulera
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), "best_gosling_model.pth")
                    print("✅ Zapisano nowy najlepszy model!")

    print(f'\n🏆 Najlepsza dokładność walidacji: {best_acc:.4f}')
    
    # Ładujemy najlepsze wagi
    model.load_state_dict(best_model_wts)
    return model

# Uruchomienie
model = train_model(model, train_loader, val_loader, EPOCHS)