import torch
from torchvision import models, transforms
from PIL import Image

# Konfiguracja
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definicje klas
model_classes = {
    'trek': ['domane', 'emonda', 'madone'],
    'canyon': ['aeroad', 'endurance', 'ultimate'],
    'specialized': ['aethos', 'roubaix', 'tarmac']
}

# Wczytaj modele
def load_model(model_path, num_classes):
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Wczytanie modeli
trek_model = load_model("models/trek/resnet50_trek_finetuned_weighted.pth", len(model_classes['trek']))
canyon_model = load_model("models/canyon/resnet50_canyon_finetuned.pth", len(model_classes['canyon']))
specialized_model = load_model("models/specialized/resnet50_specialized_finetuned.pth", len(model_classes['specialized']))

# Transformacja
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Funkcja predykcji pojedynczego modelu
def predict_single(model, classes, image):
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return classes[pred.item()], conf.item()

# Główna funkcja predykcji
def predict_combined(image_path):
    # Wczytanie i przygotowanie zdjęcia
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Predykcje z każdego modelu
    predictions = {}

    trek_pred, trek_conf = predict_single(trek_model, model_classes['trek'], image)
    predictions['trek'] = (trek_pred, trek_conf)

    canyon_pred, canyon_conf = predict_single(canyon_model, model_classes['canyon'], image)
    predictions['canyon'] = (canyon_pred, canyon_conf)

    specialized_pred, specialized_conf = predict_single(specialized_model, model_classes['specialized'], image)
    predictions['specialized'] = (specialized_pred, specialized_conf)

    # Znajdź najlepszą predykcję (największy confidence)
    best_brand, (best_model, best_confidence) = max(predictions.items(), key=lambda x: x[1][1])

    # Wyświetlenie wyniku
    print(f"🧠 Predykcja: {best_brand.upper()} {best_model.upper()} ({best_confidence * 100:.2f}%)")

# PRZYKŁAD UŻYCIA:
print("specialized roubaix")
predict_combined("test_photos/test_photos_specialized/roubaixx.png")

print("specialized aethos")
predict_combined("test_photos/test_photos_specialized/aethos.png")

print("specialized tarmac")
predict_combined("test_photos/test_photos_specialized/tarmac.png")

print("trek madone")
predict_combined("test_photos/test_photos_trek/madone2.png")

print("trek emonda")
predict_combined("test_photos/test_photos_trek/emondka.png")

print("trek domane - dobrze")
predict_combined("test_photos/test_photos_trek/domane.png")

print("trek domane - źle")
predict_combined("test_photos/test_photos_trek/domanefinal.png")

print("canyon aeroad")
predict_combined("test_photos/test_photos_canyon/aeroad.png")

print("canyon endurance")
predict_combined("test_photos/test_photos_canyon/endurance.png")

print("canyon ultimate")
predict_combined("test_photos/test_photos_canyon/ulti.png")