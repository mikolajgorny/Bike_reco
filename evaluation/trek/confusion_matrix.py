import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Klasy (w tej samej kolejności jak w ImageFolder!)
classes = ['domane', 'emonda', 'madone']
data_dir = 'dataset_split/trek/val'
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformacja – taka sama jak w walidacji
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset i DataLoader
val_dataset = datasets.ImageFolder(data_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Wczytaj model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("models/trek/resnet50_trek_finetuned_weighted.pth", map_location=device))
model = model.to(device)
model.eval()

# Predykcja na zbiorze walidacyjnym
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Tworzenie i wyświetlenie macierzy
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Zbiór walidacyjny")
plt.show()
