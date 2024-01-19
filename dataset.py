import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset


class XRayAgeDataset(Dataset):
    """Dataset for X-ray age prediction."""

    def __init__(self, csv_file, image_folder, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.image_folder, f"{self.data_frame.iloc[idx, 0]}.png"
        )
        image = Image.open(img_name)
        age = self.data_frame.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, age
