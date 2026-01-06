import os
import shutil
import random

def split_dataset(input_dir, output_dir, split_ratio=0.8):
    random.seed(42)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        for phase, image_list in zip(['train', 'val'], [train_images, val_images]):
            out_class_dir = os.path.join(output_dir, phase, class_name)
            os.makedirs(out_class_dir, exist_ok=True)
            for img_name in image_list:
                src = os.path.join(class_path, img_name)
                dst = os.path.join(out_class_dir, img_name)
                shutil.copy2(src, dst)

        print(f"{class_name}: {len(train_images)} train, {len(val_images)} val")

if __name__ == "__main__":
    input_dir = "dataset/cervelo"
    output_dir = "dataset_split/cervelo"
    split_dataset(input_dir, output_dir)
