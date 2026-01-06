import torch
from torchvision import models, transforms
from PIL import Image

# Konfiguracja
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['aethos', 'roubaix', 'tarmac']

# Transformacja identyczna jak podczas walidacji
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Wczytaj model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("models/specialized/resnet50_specialized_finetuned.pth", map_location=device))
model = model.to(device)
model.eval()

# Funkcja predykcji
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    predicted_class = classes[pred.item()]
    confidence = conf.item() * 100
    print(f"🧠 Predicted: {predicted_class.upper()} ({confidence:.2f}%)")

# Przykład użycia
predict("test_photos/test_photos_specialized/roubaix3.png")  # <- zmień ścieżkę na swój obrazek!
