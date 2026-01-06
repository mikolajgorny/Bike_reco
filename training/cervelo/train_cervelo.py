import os
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Ścieżki
train_dir = "dataset_split/cervelo/train"
val_dir = "dataset_split/cervelo/val"
model_save_path = "models/cervelo/resnet50_cervelo_finetuned.pth"

# Parametry
num_epochs = 30
batch_size = 16
num_classes = 3
learning_rate = 0.0003
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformacje
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset i DataLoader
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model: ResNet50 + nowa warstwa końcowa
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss + Optymalizator
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

# Trenowanie
best_val_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    correct = total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # Ewaluacja
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - train acc: {train_acc:.4f} - val acc: {val_acc:.4f}")

    # Zapis najlepszego modelu
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print("✅ Zapisano najlepszy model.")

print("✅ Zakończono trening. Najlepsze val_acc:", best_val_acc)
