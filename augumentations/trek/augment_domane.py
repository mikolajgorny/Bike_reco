import os
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Ścieżki
input_dir = '../../dataset/trek/domane'
output_dir = '../../dataset/trek/domane_augmented'
os.makedirs(output_dir, exist_ok=True)

# Augmentacja – lekka i realistyczna
augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

# Dla każdego zdjęcia zrób 10 wariantów
num_augmented_per_image = 10
img_extensions = ['.jpg', '.jpeg', '.png']

images = [f for f in os.listdir(input_dir) if any(f.lower().endswith(ext) for ext in img_extensions)]

print(f"🔧 Rozpoczynam augmentację {len(images)} zdjęć...")
for img_name in tqdm(images):
    img_path = os.path.join(input_dir, img_name)
    img = Image.open(img_path).convert('RGB')

    for i in range(num_augmented_per_image):
        aug_img = augmentation(img)
        aug_img.save(os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"))

print("✅ Augmentacja zakończona. Zapisano do:", output_dir)
