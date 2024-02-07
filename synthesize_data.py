import os
import pandas as pd
from PIL import Image
import random
import sys

# Define paths
dataset_dir = "dataset/train"
images_dir = os.path.join(dataset_dir, "image")
labels_csv = os.path.join(dataset_dir, "train.csv")
# Read the CSV file containing labels
labels_df = pd.read_csv(labels_csv)


# Function to add padding to an image
def add_padding(image, position):
    width, height = image.size
    max_padding = max(width, height)
    padding = random.randint(
        1, max_padding // 4
    )  # Random padding size up to 1/4 of the max dimension
    new_width = width + padding if position in ["left", "right"] else width
    new_height = height + padding if position in ["top", "bottom"] else height
    result = Image.new(image.mode, (new_width, new_height), 0)
    if position == "top":
        result.paste(image, (0, padding))
    elif position == "bottom":
        result.paste(image, (0, 0))
    elif position == "left":
        result.paste(image, (padding, 0))
    elif position == "right":
        result.paste(image, (0, 0))
    return result


# Function to rotate an image
def rotate_image(image):
    angle = random.choice([90, 180, 270])  # Random rotation angle
    return image.rotate(angle)


# Create a directory for the synthesized data
# Process each image
image_dir = sys.argv[1]

for index, row in labels_df.iterrows():
    image_id = row["id"]
    label = row["boneage"]
    male = row["male"]
    image_path = os.path.join(image_dir, f"{image_id}.png")
    image = Image.open(image_path)
    # Add padding
    positions = ["top", "bottom", "left", "right"]
    position = random.choice(positions)
    image_with_padding = add_padding(image, position)
    # Rotate image
    rotated_image = rotate_image(image_with_padding)
    # Save the synthesized image
    synthesized_image_path = os.path.join(image_dir, f"synthesized_{image_id}.png")
    rotated_image.save(synthesized_image_path)
    # Add a row to the labels dataframe
    new_row = {"id": f"synthesized_{image_id}", "boneage": label, "male": male}
    labels_df._append(new_row, ignore_index=True)

synthesized_labels_path = os.path.join(
    "/".join(image_dir.split("/")[:-1]), "train_with_synthesis.csv"
)
labels_df.to_csv(synthesized_labels_path)
print("Data synthesis complete.")
