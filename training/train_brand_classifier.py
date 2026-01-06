import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Konfiguracja
data_dir = 'dataset_brand'
batch_size = 16
num_classes = 3  # trek, canyon, specialized
num_epochs = 20
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformacje
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# Datasets i Dataloaders
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
    for x in ['train', 'val']
}
class_names = image_datasets['train'].classes
print("Klasy:", class_names)

# Model - ResNet18
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Zamrażamy feature extractor

model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Funkcja straty i optymalizacja
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = Adam(model.fc.parameters(), lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# Trening
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    for phase in ['train', 'val']:
        running_loss = 0.0
        running_corrects = 0

        model.train() if phase == 'train' else model.eval()

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])
        print(f"{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

    scheduler.step()

# Zapis modelu
os.makedirs("models/brand_classifier", exist_ok=True)
torch.save(model.state_dict(), "models/brand_classifier/resnet18_brand_classifier.pth")
print("✅ Model Brand Classifier zapisany!")
