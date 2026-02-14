import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import copy
from tqdm import tqdm

# --- KONFIGURACJA ---
TRAIN_DIR = '../dataset/train' 
VAL_DIR = '../dataset/val'
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.0001
NUM_WORKERS = 4 

# Definicja funkcji trenującej (może być tutaj lub wewnątrz main, ale tutaj jest czytelniej)
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, device, epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(epochs):
        print(f'\nEpoka {epoch+1}/{epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            loop = tqdm(dataloader, leave=True)
            
            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        preds = torch.sigmoid(outputs) > 0.5

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                loop.set_description(f"{phase.upper()}")
                loop.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), "best_gosling_model.pth")
                    print("✅ Zapisano nowy najlepszy model!")

    print(f'\n🏆 Najlepsza dokładność walidacji: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# 🔥 WAŻNE: Cała logika wykonawcza musi być w tym bloku na Windows!
if __name__ == '__main__':
    # Wykrywanie sprzętu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Używam urządzenia: {device}")
    if device.type == 'cuda':
        print(f"   Karta: {torch.cuda.get_device_name(0)}")

    # 1. Transformacje
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 2. Dataset i Loader
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)

    print(f"📂 Klasy: {train_dataset.classes}")
    print(f"📊 Dane treningowe: {len(train_dataset)} | Walidacyjne: {len(val_dataset)}")

    # 3. Model
    print("🧠 Pobieranie modelu ResNet18...")
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(device)

    # 4. Optymalizator i Loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
    scaler = torch.amp.GradScaler('cuda')

    # Uruchomienie
    # Przekazujemy teraz wszystkie obiekty do funkcji, aby uniknąć problemów z zasięgiem zmiennych
    model = train_model(
        model, train_loader, val_loader, 
        criterion, optimizer, scheduler, scaler, device, 
        EPOCHS
    )