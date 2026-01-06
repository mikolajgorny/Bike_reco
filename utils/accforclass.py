from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import classification_report

# Wybierz markę i ścieżki
brand = 'canyon'  # lub 'canyon', 'specialized'
val_dir = f'../dataset_split/{brand}/val'
model_path = f'../models/canyon/resnet50_canyon_finetuned.pth'

class_names = {
    'trek': ['domane', 'emonda', 'madone'],
    'canyon': ['aeroad', 'endurance', 'ultimate'],
    'specialized': ['aethos', 'roubaix', 'tarmac']
}[brand]

# Transformacja walidacyjna
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wczytanie modelu
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

# Loader
dataset = datasets.ImageFolder(val_dir, transform=transform)
loader = DataLoader(dataset, batch_size=16)

all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        outputs = model(x)
        preds = outputs.argmax(1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(y.tolist())

print(classification_report(all_labels, all_preds, target_names=class_names))
