import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============ KONFIGURACJA ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

brands = {
    'trek': ['domane', 'emonda', 'madone'],
    'canyon': ['aeroad', 'endurance', 'ultimate'],
    'specialized': ['aethos', 'roubaix', 'tarmac']
}

models_paths = {
    'trek': 'models/trek/resnet50_trek_finetuned_weighted.pth',
    'canyon': 'models/canyon/resnet50_canyon_finetuned.pth',
    'specialized': 'models/specialized/resnet50_specialized_finetuned.pth'
}

val_dirs = {brand: Path(f'dataset_split/{brand}/val') for brand in brands}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============ ŁADOWANIE MODELI ============
def load_model(path, num_classes):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

models_loaded = {
    brand: load_model(models_paths[brand], len(classes))
    for brand, classes in brands.items()
}

# ============ PREDYKCJA JAK W APLIKACJI ============
def predict(image_tensor):
    predictions = {}
    for brand, model in models_loaded.items():
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            predictions[f"{brand}_{brands[brand][pred.item()]}"] = conf.item()
    return max(predictions, key=predictions.get)

# ============ WALIDACJA ============
y_true, y_pred = [], []

for brand, model_list in brands.items():
    brand_dir = val_dirs[brand]
    if not brand_dir.exists():
        continue

    for model_name in model_list:
        model_path = brand_dir / model_name
        if not model_path.is_dir():
            continue

        for file in os.listdir(model_path):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            full_path = model_path / file
            try:
                image = Image.open(full_path).convert("RGB")
                tensor = transform(image).unsqueeze(0).to(device)

                pred = predict(tensor)
                true = f"{brand}_{model_name}"

                y_true.append(true)
                y_pred.append(pred)
            except Exception as e:
                print(f"Błąd przy {file}: {e}")

# ============ RAPORT & MACIERZ ============
print("\n--- Raport jakości klasyfikacji ---")
print(classification_report(y_true, y_pred, zero_division=0))

labels = sorted(set(y_true + y_pred))
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Predykcja")
plt.ylabel("Prawdziwa etykieta")
plt.title("Macierz pomyłek – logika aplikacyjna (3 modele ResNet50)")
plt.tight_layout()
plt.savefig("confusion_matrix_3x3_applogic.png")
plt.show()
