import argparse
import random
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import AgePredictionCNN  # do not remove, is actually needed by pytorch
from train import ModelHyperparameters, load_config
from dataset import XRayAgeDataset


def parse_args():
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(
        description="Inference for bone age prediction CNN."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config.json file"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to the X-ray image for prediction",
    )
    parser.add_argument("--all", action="store_true", help="Run over all images")
    parser.add_argument("--random", type=int, help="Run over random images")
    args = parser.parse_args()
    if not any([args.image, args.all, args.random]):
        raise ValueError("Please specify an image, --all or --random")
    elif (
        sum([bool(args.image), bool(args.all), bool(args.random)]) > 1
    ):  # more than 1 specified
        raise ValueError("Please specify only one of --image, --all or --random")

    return args


def load_model(model_path):
    """Load the trained model from a file."""
    model = torch.load(model_path)
    model.eval()
    return model


def predict_age(model, image_path, transform, device):
    """Predict the age from an X-ray image using the trained model."""
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image).item()
    return prediction * model.labels_norm  # Un-normalize the prediction


def load_correct_label(label_path, id):
    """Load the correct labels from the dataset using the csv library."""
    with open(label_path, "r") as f:
        for line in f.read().splitlines():
            row = line.split(",")
            if row[0] == id:
                # print(f"Found ID {id} in {label_path}, it has label {row[1]}")
                return float(row[1])
            elif not row:
                raise ValueError(f"ID {id} not found in {label_path}")


def main():
    args = parse_args()
    config = load_config(args.config)

    hyperparams = ModelHyperparameters(**config["hyperparameters"])
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["testing"]["cuda"] else "cpu"
    )
    transform = transforms.Compose(
        [
            transforms.Resize(hyperparams.image_size),
            transforms.ToTensor(),
        ]
    )
    model_path = config["testing"]["model_save_path"]
    label_path = config["testing"]["dataset_path"] + config["testing"]["csv_file"]

    model = load_model(model_path).to(device)

    if args.all:
        mean_abs_diff = 0
        image_dir = (
            config["testing"]["dataset_path"] + config["testing"]["image_folder"]
        )
        labels_file = config["testing"]["dataset_path"] + config["testing"]["csv_file"]
        dataset = XRayAgeDataset(
            csv_file=labels_file,
            image_folder=image_dir,
            transform=transform,
        )

        dataloader = DataLoader(
            dataset, batch_size=hyperparams.batch_size, shuffle=True
        )

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels
            outputs = model(inputs)
            mean_abs_diff += (outputs * model.labels_norm - labels).abs().sum().item()

        print(
            f"Mean Absolute Difference: MAD(actual, predicted) = {mean_abs_diff / len(dataset)}"
        )
    elif args.random:
        images = os.listdir(
            config["testing"]["dataset_path"] + config["testing"]["image_folder"]
        )
        ground_truths, predictions = [], []

        for _ in range(args.random):
            image_id = images[random.randrange(len(images))].split(".")[0]
            image_path = (
                config["testing"]["dataset_path"]
                + config["testing"]["image_folder"]
                + image_id
                + ".png"
            )
            ground_truth = load_correct_label(label_path, image_id)
            predicted_age = predict_age(model, image_path, transform, device)
            ground_truths.append(ground_truth)
            predictions.append(predicted_age)

        print("T      P      D")
        for t, p in zip(ground_truths, predictions):
            diff = abs(round(t) - round(p))
            print(f"{t:<7.0f}{p:<7.0f}{diff:<7.0f}")

        print("\nMean   Mean   Mean")
        mean_gt = sum(ground_truths) / len(ground_truths)
        mean_pred = sum(predictions) / len(predictions)
        mean_diff = sum(
            map(lambda gt, p: abs(round(gt - p)), ground_truths, predictions)
        ) / len(ground_truths)
        print(f"{mean_gt:<7.0f}{mean_pred:<7.0f}{mean_diff:<7.0f}")

    elif args.image:
        image_id = args.image.split("/")[-1].split(".")[0]
        ground_truth = load_correct_label(label_path, image_id)
        predicted_age = predict_age(model, args.image, transform, device)
        print(f"Predicted bone age: {predicted_age:.0f} months")
        print(f"Ground truth bone age: {ground_truth:.0f} months")
        print(f"Absolute Difference: {abs(round(ground_truth - predicted_age))} months")
    else:
        raise ValueError("Please specify an image, --all or --random")


if __name__ == "__main__":
    main()
