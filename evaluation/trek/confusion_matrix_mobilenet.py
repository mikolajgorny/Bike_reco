import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import numpy as np

# Ustawienia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['domane', 'emonda', 'madone']
batch_size = 32

# Wczytanie transformacji
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Wczytanie modelu MobileNetV2
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
model.load_state_dict(torch.load("../../mobilenetv2_trek_finetuned.pth"))
model = model.to(device)
model.eval()

# Ładowanie danych
val_dir = "../../dataset_split/trek/val"  # Zakładając, że masz folder z danymi walidacyjnymi
val_data = datasets.ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Funkcja do obliczania confusion matrix
def compute_confusion_matrix():
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Computing predictions"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm)

# Funkcja do rysowania confusion matrix
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.show()

# Uruchomienie
compute_confusion_matrix()
