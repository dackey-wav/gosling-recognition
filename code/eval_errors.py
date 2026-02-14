import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import shutil
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TEST_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'test')
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_gosling_model.pth")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "analysis_results")

FP_DIR = os.path.join(OUTPUT_DIR, "false_positives")
FN_DIR = os.path.join(OUTPUT_DIR, "false_negatives")

def setup_dirs():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(FP_DIR)
    os.makedirs(FN_DIR)
    print(f"📁 Analysis directories created: {OUTPUT_DIR}")

def analyze_errors():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Analysis on device: {device}")
    setup_dirs()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Classes: 0 -> gosling, 1 -> negative
    print(f"Classes: {dataset.class_to_idx}")

    # Model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    
    print("🔍 Scanning for errors...")

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            
            output = model(inputs)
            prob_negative = torch.sigmoid(output).item()
            
            # Class logic:
            # 0 = Gosling
            # 1 = Negative
            # prob_negative 0.99 -> Class 1 (Negative)
            
            pred_label = 1 if prob_negative > 0.5 else 0
            true_label = int(labels.item())
            
            all_preds.append(pred_label)
            all_labels.append(true_label)

            # Get path to the original file
            # dataset.samples - list of tuples (path, class_index)
            original_path = dataset.samples[i][0]
            filename = os.path.basename(original_path)

            # --- ERROR ANALYSIS ---
            # False Positive (FP): Predicted Gosling (0), but actually Negative (1)
            if pred_label == 0 and true_label == 1:
                conf_gosling = 1.0 - prob_negative
                dest = os.path.join(FP_DIR, f"conf_{conf_gosling:.2f}_{filename}")
                shutil.copy(original_path, dest)

            # False Negative (FN): Predicted Negative (1), but actually Gosling (0)
            elif pred_label == 1 and true_label == 0:
                conf_neg = prob_negative
                dest = os.path.join(FN_DIR, f"conf_{conf_neg:.2f}_{filename}")
                shutil.copy(original_path, dest)

    # Report
    print("\n📊 REPORT:")
    print(classification_report(all_labels, all_preds, target_names=['Gosling', 'Negative']))

    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Gosling', 'Negative'], yticklabels=['Gosling', 'Negative'])
    plt.title('Confusion Matrix (Errors)')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    
    print(f"✅ Analysis complete. Check the folder: {OUTPUT_DIR}")
    print(f"👀 False Positives (Model sees Gosling where there is none): {len(os.listdir(FP_DIR))}")
    print(f"👀 False Negatives (Model failed to recognize Gosling): {len(os.listdir(FN_DIR))}")

if __name__ == "__main__":
    analyze_errors()
