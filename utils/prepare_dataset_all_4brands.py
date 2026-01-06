import os
import shutil

def copy_models_from_brand(src_folder, dst_folder):
    for split in ['train', 'val']:
        split_src = os.path.join(src_folder, split)
        split_dst = os.path.join(dst_folder, split)

        for model_name in os.listdir(split_src):
            model_path = os.path.join(split_src, model_name)
            if not os.path.isdir(model_path):
                continue

            dst_model_path = os.path.join(split_dst, model_name)
            os.makedirs(dst_model_path, exist_ok=True)

            for img in os.listdir(model_path):
                src_img = os.path.join(model_path, img)
                dst_img = os.path.join(dst_model_path, img)
                shutil.copy2(src_img, dst_img)

if __name__ == "__main__":
    output_dir = "dataset_split_12class"  # folder wyjściowy ze wszystkimi modelami rowerów
    brand_folders = [
        "dataset_split/trek",
        "dataset_split/canyon",
        "dataset_split/specialized",
        "dataset_split/cervelo"
    ]

    for brand_path in brand_folders:
        copy_models_from_brand(brand_path, output_dir)

    print("✅ Zbiór danych 12-klasowy (modele rowerów) przygotowany!")
