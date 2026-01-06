import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Ustawienia treningu
num_epochs = 20
batch_size = 32
learning_rate = 0.001

# Ustawienia ścieżek do danych
train_dir = "../../dataset_split/trek/train"
val_dir = "../../dataset_split/trek/val"

# Przygotowanie augmentacji i transformacji
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Załadowanie zbioru treningowego i walidacyjnego
train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Załadowanie pretrenowanego modelu MobileNetV2
model = models.mobilenet_v2(pretrained=True)

# Zamrożenie wag w początkowych warstwach
for param in model.parameters():
    param.requires_grad = False

# Zmiana klasyfikatora na 3 klasy
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)

# Przeniesienie modelu na GPU, jeśli dostępne
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Definicja funkcji straty i optymalizatora
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Funkcja do treningu modelu
def train_model():
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Trening
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Obliczanie statystyk
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Zapis wyników
        train_acc = 100 * correct / total
        print(f"Train Loss: {running_loss / len(train_loader):.4f} | Train Accuracy: {train_acc:.2f}%")

        # Walidacja
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Obliczanie statystyk
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Validation Loss: {val_loss / len(val_loader):.4f} | Validation Accuracy: {val_acc:.2f}%")

        # Zapis modelu po każdej epoce
        torch.save(model.state_dict(), "mobilenetv2_trek_finetuned.pth")
        print("Model saved!")

# Rozpoczęcie treningu
train_model()
