import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# =======================
# PARAMETRY
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
epochs = 15
learning_rate = 0.0005
num_classes = 3

# =======================
# ŚCIEŻKI DO DANYCH
# =======================
train_dir = "dataset_split/specialized/train"
val_dir = "dataset_split/specialized/val"

# =======================
# TRANSFORMACJE
# =======================
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =======================
# ŁADOWANIE DANYCH
# =======================
train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

print("KLASY:", train_dataset.classes)

# =======================
# MODEL: EfficientNet-B0
# =======================
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
for param in model.features.parameters():
    param.requires_grad = False  # zamrażamy feature extractor

# Podmieniamy warstwę końcową
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# =======================
# STRATEGIA TRENINGU
# =======================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# =======================
# PĘTLA TRENINGOWA
# =======================
def train():
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        # Walidacja
        model.eval()
        val_loss, val_correct = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

                all_preds += outputs.argmax(1).cpu().tolist()
                all_labels += labels.cpu().tolist()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} - train acc: {train_acc:.4f} - val acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/specialized/efficientnet_specialized.pth")
            print("✅ Zapisano najlepszy model.")

    print("\n--- Raport końcowy ---")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

if __name__ == "__main__":
    train()
