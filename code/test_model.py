import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- PATH SETTINGS ---
# Get path to the folder containing THIS script (code/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # Parent directory (gosling-recognition/)

# Build full paths
TEST_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'test')
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_gosling_model.pth")
IMG_SIZE = 224
BATCH_SIZE = 32

# Hardware detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Testing on: {device}")
print(f"📁 Test directory: {TEST_DIR}")
print(f"💾 Model: {MODEL_PATH}")

if not os.path.exists(TEST_DIR):
    print("❌ Error: dataset/test doesn\'t exist! Run split-dataset.py first.")
    exit()

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model file doesn\'t exist: {MODEL_PATH}")
    exit()

# 1. Transformations (Identical to training ResNet)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 2. Loading data
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Building the ResNet18 model
print("🧠 Building model ResNet18...")
model = models.resnet18(weights=None) 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1) 
model = model.to(device)

# 4. Loading weights
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✅ Loaded weights from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading weights: {e}")
    exit()

model.eval()

# 5. Large test
all_preds = []
all_labels = []

print(f"Starting test on {len(test_dataset)} images...")

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5 
        
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.numpy())

# 6. Results and Report
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - ResNet18')

plt.savefig('result_resnet.png')
print("Saved as 'result_resnet.png'")