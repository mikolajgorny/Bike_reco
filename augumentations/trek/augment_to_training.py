import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Ścieżki wejścia i wyjścia
INPUT_PATHS = {
    "domane": "augment_input/domane",
    "emonda": "augment_input/emonda"
}

OUTPUT_PATH = "../../dataset_split/trek/train"

# Konfiguracja augmentacji
AUGMENT_COUNT = 30
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
])

# Funkcja augmentująca i zapisująca
def augment_images(label):
    input_dir = INPUT_PATHS[label]
    output_dir = os.path.join(OUTPUT_PATH, label)
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir), desc=f"Augmenting {label}"):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert("RGB")

        for i in range(AUGMENT_COUNT):
            augmented = transform(img)
            out_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_aug{i}.jpg")
            augmented.save(out_path)

# Augmentuj obie klasy
augment_images("domane")
augment_images("emonda")
