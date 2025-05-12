import os
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Parameters
IMG_SIZE = 32
DATASET_DIR = "feral cats dataset"
OUTPUT_DIR = "processed_dataset"
TEST_RATIO = 1/6

# Helper: center crop with padding
def center_crop_and_resize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # fallback

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    cropped = img[y:y+h, x:x+w]
    size = max(w, h)
    square = np.zeros((size, size, 3), dtype=np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    return cv2.resize(square, (IMG_SIZE, IMG_SIZE))

# Save a batch of images
def save_batch(images, label, split):
    base_dir = os.path.join(OUTPUT_DIR, split, label)
    os.makedirs(base_dir, exist_ok=True)
    existing = len(os.listdir(base_dir)) if os.path.exists(base_dir) else 0
    for i, img in enumerate(images):
        filename = f"{label}_{existing + i}.png"
        filepath = os.path.join(base_dir, filename)
        cv2.imwrite(filepath, img)

# Process per-class to avoid memory overload
print("Processing and saving class-wise batches...")
class_image_paths = defaultdict(list)

# Phase 1: Collect filepaths grouped by class
for root, _, files in os.walk(DATASET_DIR):
    for file in files:
        if file.lower().endswith(".jpg"):
            label = os.path.basename(root)
            class_image_paths[label].append(os.path.join(root, file))

# Phase 2: For each class, read and save in batches
for label, filepaths in tqdm(class_image_paths.items(), desc="Classes"):
    # Shuffle and split into train/test
    train_files, test_files = train_test_split(filepaths, test_size=TEST_RATIO, random_state=42)

    for split_name, files in [("train", train_files), ("test", test_files)]:
        batch = []
        for filepath in files:
            try:
                with Image.open(filepath) as pil_img:
                    pil_img = pil_img.convert("RGB")
                    img = np.array(pil_img)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Skipping file {filepath}: {e}")
                continue

            processed = center_crop_and_resize(img)
            batch.append(processed)

            if len(batch) >= 128:
                save_batch(batch, label, split_name)
                batch = []

            if batch:
                save_batch(batch, label, split_name)

print("✅ Done. All images processed and saved.")
