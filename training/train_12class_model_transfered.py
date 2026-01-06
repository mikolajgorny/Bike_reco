import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# === PARAMETRY ===
data_dir = "dataset_split_12class"
model_pretrained_path = "models/unified/resnet50_allbrands.pth"  # <- model 9-klasowy
model_save_path = "models/unified/resnet50_12class_transfer.pth"
image_size = 224
batch_size = 32
num_epochs = 15
learning_rate = 1e-4

# === TRANSFORMACJE ===
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === DANE ===
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)

# === MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 9)  # tymczasowo 9, by załadować checkpoint
model.load_state_dict(torch.load(model_pretrained_path, map_location=device))

# Zamień warstwę fc na nową dla 12 klas
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# === STRATA I OPTYMIZATOR ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# === TRENING ===
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    train_correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    train_acc = train_correct / total

    # === WALIDACJA ===
    model.eval()
    val_corrects = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels).item()
            val_total += labels.size(0)

    val_acc = val_corrects / val_total
    print(f"Epoch {epoch+1}/{num_epochs} - train acc: {train_acc:.4f} - val acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_save_path)
        print("✅ Zapisano najlepszy model.")

print(f"\n✅ Zakończono trening. Najlepsze val_acc: {best_val_acc:.3f}")
