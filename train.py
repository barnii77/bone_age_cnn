import argparse
import dataclasses
import json
import os
from typing import Any
from PIL import Image
import pandas as pd
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from model import AgePredictionCNN


@dataclasses.dataclass
class ModelHyperparameters:
    """Hyperparameters for the CNN model."""

    learning_rate: float
    batch_size: int
    num_epochs: int
    image_size: tuple
    num_channels: int
    dropout_rate: float
    kernel_size: int
    stride: int
    padding: int
    labels_norm: float  # factor to normalize labels into range(0, 1) (divide by this factor)


@dataclasses.dataclass
class TrainingConfig:
    """Configuration for training the model."""

    dataset_path: str
    csv_file: str
    image_folder: str
    model_save_path: str
    cuda: bool


def load_config(config_path: str) -> Any:
    """Load configuration from a JSON file."""
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config


def parse_args() -> Any:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a CNN on X-ray images for age prediction."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config.json file"
    )
    args = parser.parse_args()
    return args


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


def train_model(
    model,
    dataloader,
    criterion,
    optimizer,
    num_epochs,
    device,
    training_config,
    hyperparams,
):
    """Train the model."""
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels / hyperparams.labels_norm
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print("DEBUG: output std = {outputs.std()}")
            print(f"Epoch {epoch+1}/{num_epochs}, loss: {loss.detach().item():.4f}")
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Epoch loss: {epoch_loss:.4f}")
        torch.save(model.state_dict(), training_config.model_save_path)
        print("Model checkpointed.")


def main():
    """Main function to run the training."""
    args = parse_args()
    config = load_config(args.config)
    hyperparams = ModelHyperparameters(**config["hyperparameters"])
    training_config = TrainingConfig(**config["training"])
    transform = transforms.Compose(
        [
            transforms.Resize(hyperparams.image_size),
            transforms.ToTensor(),
        ]
    )
    dataset = XRayAgeDataset(
        csv_file=os.path.join(training_config.dataset_path, training_config.csv_file),
        image_folder=os.path.join(
            training_config.dataset_path, training_config.image_folder
        ),
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=hyperparams.batch_size, shuffle=True)
    model = AgePredictionCNN(
        hyperparams.num_channels,
        hyperparams.dropout_rate,
        hyperparams.kernel_size,
        hyperparams.stride,
        hyperparams.padding,
        hyperparams.labels_norm,
    )
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=hyperparams.learning_rate)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and training_config.cuda else "cpu"
    )
    train_model(
        model,
        dataloader,
        criterion,
        optimizer,
        hyperparams.num_epochs,
        device,
        training_config,
        hyperparams,
    )
    torch.save(model, training_config.model_save_path)
    print("Model saved + training completed.")


if __name__ == "__main__":
    main()
