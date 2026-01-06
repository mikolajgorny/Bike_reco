import torch
from torchvision import models, transforms
from PIL import Image

# Konfiguracja
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['domane', 'emonda', 'madone']  # Nazwy klas

# Przygotowanie obrazu (musi być tak samo przetwarzany jak w walidacji!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Wczytaj model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("models/trek/resnet50_trek_finetuned_weighted.pth", map_location=device))
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
predict("test_photos/test_photos_trek/domane1.png")  # <- Podmień nazwę pliku, jeśli inna
#predict("emonda1.png")  # <- Podmień nazwę pliku, jeśli inna
#predict("madone1.png")  # <- Podmień nazwę pliku, jeśli inna
