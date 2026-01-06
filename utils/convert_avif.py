import os
import cv2

input_folder = "dataset_split/trek/val/domane"

for filename in os.listdir(input_folder):
    if filename.endswith(".avif"):
        avif_path = os.path.join(input_folder, filename)
        jpg_path = os.path.join(input_folder, filename.replace(".avif", ".jpg"))

        img = cv2.imread(avif_path)
        if img is not None:
            cv2.imwrite(jpg_path, img)
            os.remove(avif_path)
            print(f"✅ Zmieniono: {filename} -> {jpg_path}")
        else:
            print(f"⚠️ Nie udało się otworzyć: {filename}")
