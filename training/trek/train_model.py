import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Konfiguracja
data_dir = '../../dataset_split'
batch_size = 16
num_classes = 3
num_epochs = 30
lr = 5e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Zaawansowana augmentacja i preprocessing
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))
        ], p=0.5),
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

# Dataset i DataLoader
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
    for x in ['train', 'val']
}

# Model ResNet50 + fine-tuning
model = models.resnet50(pretrained=True)
for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Podmiana końcowej warstwy
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Domane: 1.5x ważniejsza, Madone: 1.2x, Émonda: 1.0 (bo radzi sobie świetnie)
class_weights = torch.tensor([1.5, 1.0, 1.2]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)


# Optymalizator + scheduler
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# Trening
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
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

#Model będzie zapisany dopiero, gdy się poprawi
torch.save(model.state_dict(), 'resnet50_trek_finetuned_weighted.pth')
print("✅ Model saved!")
