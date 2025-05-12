import os
import cv2
import numpy as np
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
from ultralytics import YOLO

# Allow truncated JPEGs
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Settings
IMG_SIZE = 32
DATASET_DIR = "feral cats dataset"
OUTPUT_DIR = "processed_dataset_obj_detection"
TEST_RATIO = 1/6
BATCH_SIZE = 128
CAT_CLASS_ID = 15  # COCO class for cat

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # nano model for speed

# Preprocess with YOLO detection
def detect_cat_and_crop(img):
    results = model(img, verbose=False)[0]
    cats = [b for b in results.boxes if int(b.cls) == CAT_CLASS_ID]

    if not cats:
        return cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # fallback

    x1, y1, x2, y2 = map(int, cats[0].xyxy[0])  # Use first detected cat
    cropped = img[y1:y2, x1:x2]
    h, w = cropped.shape[:2]
    size = max(h, w)
    square = np.zeros((size, size, 3), dtype=np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    return cv2.resize(square, (IMG_SIZE, IMG_SIZE))

# Save a batch
def save_batch(images, label, split):
    base_dir = os.path.join(OUTPUT_DIR, split, label)
    os.makedirs(base_dir, exist_ok=True)
    existing = len(os.listdir(base_dir)) if os.path.exists(base_dir) else 0
    for i, img in enumerate(images):
        filename = f"{label}_{existing + i}.png"
        cv2.imwrite(os.path.join(base_dir, filename), img)

# Phase 1: collect image paths per class
class_image_paths = defaultdict(list)
for root, _, files in os.walk(DATASET_DIR):
    for file in files:
        if file.lower().endswith(".jpg"):
            label = os.path.basename(root)
            class_image_paths[label].append(os.path.join(root, file))

# Phase 2: process class by class
for label, filepaths in tqdm(class_image_paths.items(), desc="Processing Classes"):
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

            try:
                processed = detect_cat_and_crop(img)
            except Exception as e:
                print(f"Detection failed for {filepath}: {e}")
                continue

            batch.append(processed)
            if len(batch) >= BATCH_SIZE:
                save_batch(batch, label, split_name)
                batch = []

        if batch:
            save_batch(batch, label, split_name)

print("✅ Preprocessing complete. Data saved to:", OUTPUT_DIR)
