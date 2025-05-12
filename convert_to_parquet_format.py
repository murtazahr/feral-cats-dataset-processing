import os
from glob import glob

import cv2
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Settings (matching your original script)
INPUT_DIR = "processed_dataset_obj_detection"
OUTPUT_DIR = "feral_cats"
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

def create_readme(input_dir, output_dir, img_size):
    """
    Create a comprehensive README.md file for the Hugging Face dataset card
    following the standard Hugging Face format

    Args:
        input_dir: Original dataset directory
        output_dir: Parquet output directory
        img_size: Size of the images (width/height)
    """
    # Get class information
    class_mapping_path = os.path.join(output_dir, "train_class_mapping.txt")
    classes = []

    with open(class_mapping_path, "r") as f:
        for line in f:
            class_id, class_name = line.strip().split(",")
            classes.append(class_name)

    # Count total images
    train_count = len(glob(os.path.join(input_dir, "train", "*", "*.png")))
    test_count = len(glob(os.path.join(input_dir, "test", "*", "*.png")))

    # Build the class names string with proper indentation
    class_names_str = ""
    for i, class_name in enumerate(classes):
        class_names_str += f"\n            '{i}': '{class_name}'"

    # Create the YAML metadata section
    # Modified to match the exact column structure of the parquet files
    yaml_metadata = f"""---
annotations_creators:
  - expert-generated
language_creators:
  - found
language:
  - en
license:
  - mit
multilinguality:
  - monolingual
size_categories:
  - {'1K<n<10K' if train_count + test_count < 10000 else '10K<n<100K' if train_count + test_count < 100000 else '100K<n<1M'}
source_datasets:
  - original
task_categories:
  - image-classification
task_ids:
  - multi-class-image-classification
pretty_name: Feral Cats Image Dataset
paperswithcode_id: feral-cats-identification
---
"""

    # Generate README content
    readme_content = yaml_metadata + f"""
# Dataset Card for Feral Cats Image Dataset

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Point of Contact:** mrangwala@student.unimelb.edu.au

### Dataset Summary

The Feral Cats Image Dataset consists of {train_count + test_count} {img_size}x{img_size} RGB images of feral cats categorized into {len(classes)} individual cats. The dataset was gathered from 938 trap camera sites located in the Great Otway National Park and Otway Forest Park, in Victoria, Australia. There are {train_count} images in the training dataset and {test_count} images in the test dataset.

The dataset has been pre-processed to focus on the cats by centering and cropping each image around the cat, then resizing to {img_size}x{img_size} pixels. The final format is similar to MNIST, with each image flattened into a 1D array of pixel values for efficient storage and processing.

Feral cats are a menace to native species in Australia, and this dataset helps ecologists monitor and count individual cats to understand their abundance.

### Supported Tasks and Leaderboards

- `image-classification`: The goal of this task is to classify a given image of a feral cat into one of {len(classes)} classes representing unique individual cats. This task is essential for monitoring feral cat populations without capturing or disturbing them.

### Languages

English

## Dataset Structure

### Data Instances

A data point comprises an image and its label:

```
{{
  'label': 0,  # Corresponds to a unique individual cat
  'class_name': '{classes[0] if classes else "example_cat_id"}',
  'pixel_0': 127,
  'pixel_1': 142,
  ...
  'pixel_{img_size*img_size*3-1}': 183
}}
```

### Data Fields

- `label`: An integer between 0 and {len(classes)-1} representing the unique individual cat
- `class_name`: String identifier of the individual cat
- `pixel_0` to `pixel_{img_size*img_size*3-1}`: Individual RGB pixel values (integers from 0-255) representing the flattened image

### Data Splits

The data is split into training and test sets with a 5:1 ratio, ensuring that each class (individual cat) is represented proportionally in both sets. The training set contains {train_count} images and the test set {test_count} images.

## Dataset Creation

### Curation Rationale

Individual feral cat identification is a challenging task due to insufficient and imbalanced data captured by motion cameras. This dataset was created to support the development of deep learning models that can identify individual feral cats from camera trap images, which is essential for ecological monitoring and conservation efforts. The ability to accurately identify and count feral cats is crucial for understanding their impact on native wildlife and developing effective management strategies.

### Source Data

#### Initial Data Collection and Normalization

The feral cat image data for individual cat identification was gathered from 938 trap camera sites located in the Great Otway National Park and Otway Forest Park, in Victoria, Australia (38.42 °S, 142.24 °E). At each trap site, a sensing camera was installed with infrared flash and temperature-in-motion detector, which was triggered to capture five consecutive photographs when the camera detected the movement of nearby animals.

The majority of cameras used were Reconyx Hyperfire HC600, while a small proportion consisted of PC900's HF2X's infrared camera. The raw images were pre-processed to center the cat in each photograph and cropped accordingly. All cropped images were then resized to a uniform size of {img_size}x{img_size} pixels using 0-padding to ensure consistency across the dataset.

#### Who are the source language producers?

The images were collected automatically by motion-triggered cameras in a natural wildlife setting. The cameras were set up by researchers from the University of Melbourne as part of ecological studies in the Great Otway National Park and Otway Forest Park, Victoria, Australia.

### Annotations

#### Annotation process

Manual data processing (labelling of unique cats) was done by at least two independent observers to ensure accuracy. Based on comparison of the unique markings of feral cats, each individual cat was assigned a unique identifier for later model training and identification.

#### Who are the annotators?

The annotators were researchers and experts in wildlife ecology who manually reviewed and classified each cat image based on their unique markings.

### Personal and Sensitive Information

This dataset contains only images of feral cats and does not include any personal or sensitive information.

## Considerations for Using the Data

### Social Impact of Dataset

This dataset can help improve the monitoring of feral cat populations, which is crucial for wildlife conservation efforts in Australia where feral cats are a significant threat to native species. Better identification and tracking of individual cats can lead to more effective management strategies and potentially reduce the negative impact of feral cats on biodiversity.

### Discussion of Biases

The dataset has a highly skewed class distribution, with some individual cats having significantly more images than others. This imbalance could potentially lead to biased models that perform better on classes with more samples. Additionally, the images were collected from specific regions in Victoria, Australia, and might not represent the full diversity of feral cat populations across different environments.

### Other Known Limitations

Many images in the dataset are of low quality, including some over-exposed or blurry photographs caused by animal movement. The cameras typically use burst mode once triggered and take a series of photographs in rapid succession, resulting in a large number of images with similar backgrounds and minimal changes in the cat's posture.

## Additional Information

### Dataset Curators

This dataset was curated by researchers from the Faculty of Engineering and Information Technology at The University of Melbourne, including Zihan Yang, Richard O. Sinnott, James Bailey, and Krista A. Ehinger.

### Licensing Information

cc-by-nc-nd-4.0

## Usage Example

```python
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
train_data = pq.read_table('cat_dataset_parquet/train.parquet').to_pandas()
test_data = pq.read_table('cat_dataset_parquet/test.parquet').to_pandas()

# Extract features and labels
X_train = train_data.iloc[:, 2:].values  # All pixel columns
y_train = train_data['label'].values

X_test = test_data.iloc[:, 2:].values
y_test = test_data['label'].values

# Reshape images for visualization (from 1D to 3D)
def visualize_image(pixel_values, img_size={img_size}):
    # Reshape from flat array to 3D image (height, width, channels)
    img = pixel_values.reshape(img_size, img_size, 3)
    # Convert to uint8 if necessary
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Example: Visualize the first image
visualize_image(X_train[0])
```
"""

    # Also create a dataset_card.md (for backward compatibility)
    with open(os.path.join(output_dir, "dataset_card.md"), "w") as f:
        f.write(readme_content)

    # Write README.md
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme_content)

    print(f"✅ README.md and dataset_card.md created in {output_dir}")

def create_yaml_metadata(output_dir, classes, train_count, test_count):
    """
    Create a separate metadata.yaml file which is often used by Hugging Face

    Args:
        output_dir: Parquet output directory
        classes: List of class names
        train_count: Number of training images
        test_count: Number of test images
    """
    # Build proper class_names string with careful indentation
    class_names_str = ""
    for i, class_name in enumerate(classes):
        class_names_str += f"\n            '{i}': '{class_name}'"

    # Modified to match the actual data schema instead of defining image type
    yaml_content = f"""annotations_creators:
  - expert-generated
language_creators:
  - found
language:
  - en
license:
  - mit
multilinguality:
  - monolingual
size_categories:
  - {'1K<n<10K' if train_count + test_count < 10000 else '10K<n<100K' if train_count + test_count < 100000 else '100K<n<1M'}
source_datasets:
  - original
task_categories:
  - image-classification
task_ids:
  - multi-class-image-classification
pretty_name: Feral Cats Image Dataset
dataset_info:
  config_name: feral-cats
  features:
    - name: label
      dtype: int64
    - name: class_name
      dtype: string
    - name: pixels
      sequence: 
        dtype: int64
        length: {IMG_SIZE*IMG_SIZE*3}
  splits:
    - name: train
      num_examples: {train_count}
    - name: test
      num_examples: {test_count}
configs:
  - config_name: feral-cats
    data_files:
      - split: train
        path: train.parquet
      - split: test
        path: test.parquet
    default: true
"""
    with open(os.path.join(output_dir, "metadata.yaml"), "w") as f:
        f.write(yaml_content)

    print(f"✅ metadata.yaml created in {output_dir}")

# Create a simple load script that helps Hugging Face understand your data format
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

# Count total images
train_count = len(glob(os.path.join(INPUT_DIR, "train", "*", "*.png")))
test_count = len(glob(os.path.join(INPUT_DIR, "test", "*", "*.png")))

# Create the README.md file for Hugging Face
create_readme(INPUT_DIR, OUTPUT_DIR, IMG_SIZE)

# Create additional metadata.yaml file
create_yaml_metadata(OUTPUT_DIR, sorted(train_classes.keys()), train_count, test_count)

# Create a simple .gitattributes file to tell Hugging Face how to handle the parquet files
with open(os.path.join(OUTPUT_DIR, ".gitattributes"), "w") as f:
    f.write("*.parquet filter=lfs diff=lfs merge=lfs -text\n")

print(f"✅ Conversion complete. All files saved to: {OUTPUT_DIR}")
print("Note: The metadata has been simplified to avoid schema conflicts.")
print("Hugging Face will automatically infer the schema from your parquet files.")
print("Make sure to use Git LFS when uploading .parquet files to Hugging Face.")