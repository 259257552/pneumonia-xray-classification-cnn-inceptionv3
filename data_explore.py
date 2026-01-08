import os
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = "chest_xray"
TRAIN_DIR = os.path.join(BASE_DIR, "train")

def count_images(path):
    counts = {}
    for cls in os.listdir(path):
        cls_dir = os.path.join(path, cls)
        counts[cls] = len(os.listdir(cls_dir))
    return counts

def show_sample(path, cls):
    img_name = os.listdir(os.path.join(path, cls))[0]
    img = Image.open(os.path.join(path, cls, img_name))
    plt.imshow(img, cmap="gray")
    plt.title(f"{cls} sample")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    print("Training data distribution:")
    print(count_images(TRAIN_DIR))

    show_sample(TRAIN_DIR, "NORMAL")
    show_sample(TRAIN_DIR, "PNEUMONIA")
