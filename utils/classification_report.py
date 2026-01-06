import torch
import os
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ======================
# ŚCIEŻKI
# ======================
model_path = "models/unified/resnet50_allbrands.pth"
val_dir = "dataset_split_9class/val"

# ======================
# KLASY (3 marki × 3 modele)
# ======================
all_classes = [
    "aeroad", "endurance", "ultimate",      # Canyon
    "domane", "emonda", "madone",            # Trek
    "aethos", "roubaix", "tarmac"            # Specialized
]

class_to_idx = {cls: i for i, cls in enumerate(all_classes)}

# ======================
# TRANSFORMACJE
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================
# WCZYTANIE MODELU
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(all_classes))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ======================
# PREDYKCJE
# ======================
y_true = []
y_pred = []

with torch.no_grad():
    for cls in os.listdir(val_dir):
        cls_path = os.path.join(val_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        for img_name in os.listdir(cls_path):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue

            img_path = os.path.join(cls_path, img_name)
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

            y_pred.append(predicted.item())
            y_true.append(class_to_idx[cls])

# ======================
# RAPORT KLASYFIKACJI
# ======================
print("\n--- Raport jakości klasyfikacji (9 klas) ---\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=all_classes,
    digits=2
))

# ======================
# MACIERZ POMYŁEK
# ======================
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(9, 9))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=all_classes
)
disp.plot(
    cmap="Blues",
    xticks_rotation=45,
    ax=ax,
    colorbar=True
)

plt.title("Macierz pomyłek – ResNet50 (9 klas)")
plt.tight_layout()
plt.show()
