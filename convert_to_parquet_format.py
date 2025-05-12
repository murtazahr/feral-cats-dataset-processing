import os
from glob import glob

import cv2
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Settings (matching your original script)
INPUT_DIR = "processed_dataset_obj_detection"
OUTPUT_DIR = "cat_dataset_parquet"
IMG_SIZE = 32

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_split(split):
    # List to store all image data and labels
    all_data = []
    class_mapping = {}

    # Get all class directories
    class_dirs = sorted([d for d in os.listdir(os.path.join(INPUT_DIR, split))
                         if os.path.isdir(os.path.join(INPUT_DIR, split, d))])

    # Create class mapping (class name to integer)
    for i, class_name in enumerate(class_dirs):
        class_mapping[class_name] = i

    # Save class mapping
    with open(os.path.join(OUTPUT_DIR, f"{split}_class_mapping.txt"), "w") as f:
        for class_name, class_id in class_mapping.items():
            f.write(f"{class_id},{class_name}\n")

    # Process each class
    for class_name in tqdm(class_dirs, desc=f"Processing {split} split"):
        class_id = class_mapping[class_name]

        # Get all image files for this class
        image_files = glob(os.path.join(INPUT_DIR, split, class_name, "*.png"))

        for img_path in image_files:
            # Read the image
            img = cv2.imread(img_path)

            # Convert to flattened pixel array (similar to MNIST format)
            # For RGB images, flatten each channel
            # Shape becomes (IMG_SIZE*IMG_SIZE*3,)
            flattened = img.reshape(-1)

            # Create a record for this image
            record = {
                'label': class_id,
                'class_name': class_name
            }

            # Add pixel values to the record
            # For MNIST-like format, add each pixel as a separate column
            for i, pixel_val in enumerate(flattened):
                record[f'pixel_{i}'] = int(pixel_val)

            all_data.append(record)

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Convert to PyArrow Table and save as Parquet
    table = pa.Table.from_pandas(df)
    pq.write_table(table, os.path.join(OUTPUT_DIR, f"{split}.parquet"))

    print(f"✅ {split} split saved with {len(df)} images across {len(class_dirs)} classes")

    return class_mapping

# Process train and test splits
train_classes = process_split("train")
test_classes = process_split("test")

# Verify classes are consistent
assert set(train_classes.keys()) == set(test_classes.keys()), "Train and test class mismatch!"

# Create metadata file
with open(os.path.join(OUTPUT_DIR, "dataset_info.txt"), "w") as f:
    f.write(f"Image size: {IMG_SIZE}x{IMG_SIZE}\n")
    f.write(f"Number of classes: {len(train_classes)}\n")
    f.write(f"Classes: {', '.join(sorted(train_classes.keys()))}\n")
    f.write(f"RGB image data flattened to {IMG_SIZE*IMG_SIZE*3} features\n")

print(f"✅ Conversion complete. Parquet files saved to: {OUTPUT_DIR}")
