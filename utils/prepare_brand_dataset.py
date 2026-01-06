import os
import shutil
import random

def copy_images(src_folder, dst_folder, brand_name):
    src_train = os.path.join(src_folder, 'train')
    src_val = os.path.join(src_folder, 'val')

    dst_train = os.path.join(dst_folder, 'train', brand_name)
    dst_val = os.path.join(dst_folder, 'val', brand_name)

    os.makedirs(dst_train, exist_ok=True)
    os.makedirs(dst_val, exist_ok=True)

    for src_dir, dst_dir in [(src_train, dst_train), (src_val, dst_val)]:
        for class_folder in os.listdir(src_dir):
            class_path = os.path.join(src_dir, class_folder)
            if not os.path.isdir(class_path):
                continue

            for img_name in os.listdir(class_path):
                src_img = os.path.join(class_path, img_name)
                dst_img = os.path.join(dst_dir, img_name)
                shutil.copy2(src_img, dst_img)

if __name__ == "__main__":
    output_dir = "dataset_brand"
    sources = {
        "trek": "dataset_split/trek",
        "canyon": "dataset_split/canyon",
        "specialized": "dataset_split/specialized"
    }

    for brand, path in sources.items():
        copy_images(path, output_dir, brand)

    print("✅ Brand dataset prepared!")
