import os
import torch
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===== KONFIGURACJA =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_paths = {
    'trek': 'models/trek/resnet50_trek_finetuned_weighted.pth',
    'canyon': 'models/canyon/resnet50_canyon_finetuned.pth',
    'specialized': 'models/specialized/resnet50_specialized_finetuned.pth',
    'cervelo': 'models/cervelo/resnet50_cervelo_finetuned.pth'
}

model_classes = {
    'trek': ['domane', 'emonda', 'madone'],
    'canyon': ['aeroad', 'endurance', 'ultimate'],
    'specialized': ['aethos', 'roubaix', 'tarmac'],
    'cervelo': ['s5', 'r5', 'soloist']
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== FUNKCJE =====
def load_model(model_path, num_classes):
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_single(model, classes, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return classes[pred.item()], conf.item()

# ===== WCZYTANIE MODELI =====
models_loaded = {
    brand: load_model(path, len(classes))
    for brand, (path, classes) in zip(model_paths.keys(), zip(model_paths.values(), model_classes.values()))
}

# ===== PRZETWARZANIE OBRAZÓW Z VALIDATION SET =====
all_preds = []
all_labels = []

for brand in model_classes.keys():
    val_dir = f'dataset_split/{brand}/val'
    for model_name in os.listdir(val_dir):
        model_path = os.path.join(val_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        for fname in os.listdir(model_path):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue

            fpath = os.path.join(model_path, fname)
            image = Image.open(fpath).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            best_brand = None
            best_model = None
            best_conf = -1

            for b, model in models_loaded.items():
                pred_model, conf = predict_single(model, model_classes[b], image_tensor)
                if conf > best_conf:
                    best_brand = b
                    best_model = pred_model
                    best_conf = conf

            pred_label = f"{best_brand}_{best_model}"
            true_label = f"{brand}_{model_name}"

            all_preds.append(pred_label)
            all_labels.append(true_label)

# ===== METRYKI =====
unique_labels = sorted(set(all_labels + all_preds))
cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)

plt.figure(figsize=(12, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Macierz pomyłek – logika aplikacyjna (4 marki, ResNet50)")
plt.tight_layout()
plt.savefig("confusion_matrix_4x3_applogic.png")
plt.show()

print("\n--- Raport jakości klasyfikacji ---")
print(classification_report(all_labels, all_preds, labels=unique_labels))
