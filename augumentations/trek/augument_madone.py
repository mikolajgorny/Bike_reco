import os
import random
from PIL import Image
import torchvision.transforms as transforms
import torch

# Foldery wejściowe i wyjściowe
input_folder = 'dataset_split/train/madone'  # Folder z oryginalnymi zdjęciami Madone
output_folder = 'dataset_split/train/madone_augmented'  # Folder na nowe zaugumentowane zdjęcia

# Tworzymy folder wyjściowy, jeśli nie istnieje
os.makedirs(output_folder, exist_ok=True)

# Augmentacje: losowe przycięcia, rotacje, flip, zmiana kontrastu
transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

# Liczba zaugumentowanych zdjęć
num_augmented_images = 100  # Zmienna liczba obrazów do augmentacji (np. 100)

# Funkcja do augmentacji
def augment_images():
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for i in range(num_augmented_images):
        # Wybór losowego zdjęcia z folderu
        image_name = random.choice(images)
        img_path = os.path.join(input_folder, image_name)
        img = Image.open(img_path).convert("RGB")

        # Zastosowanie augmentacji
        augmented_image = transform(img)

        # Przekształcenie tensora do obrazu
        augmented_image = transforms.ToPILImage()(augmented_image)

        # Zapisanie zaugumentowanego zdjęcia
        augmented_image.save(os.path.join(output_folder, f"augmented_{i}.jpg"))
        print(f"Augmentacja {i + 1}/{num_augmented_images} zakończona.")

# Uruchamiamy augmentację
augment_images()
print("Augmentacja zakończona. Wszystkie zdjęcia zapisane w:", output_folder)
