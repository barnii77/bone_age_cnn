import argparse
from PIL import Image
import torch
from torchvision import transforms
from train import AgePredictionCNN, ModelHyperparameters, load_config


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
        required=True,
        help="Path to the X-ray image for prediction",
    )
    return parser.parse_args()


def load_model(model_path, hyperparams):
    """Load the trained model from a file."""
    model = AgePredictionCNN(
        hyperparams.num_channels,
        hyperparams.dropout_rate,
        hyperparams.kernel_size,
        hyperparams.stride,
        hyperparams.padding,
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_age(model, image_path, transform, device):
    """Predict the age from an X-ray image using the trained model."""
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image).item()
    return prediction


def main():
    args = parse_args()
    config = load_config(args.config)
    hyperparams = ModelHyperparameters(**config["hyperparameters"])
    model_path = config["training"]["model_save_path"]
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["training"]["cuda"] else "cpu"
    )
    transform = transforms.Compose(
        [
            transforms.Resize(hyperparams.image_size),
            transforms.ToTensor(),
        ]
    )
    model = load_model(model_path, hyperparams).to(device)
    predicted_age = predict_age(model, args.image, transform, device)
    print(f"Predicted bone age: {predicted_age:.2f} months")


if __name__ == "__main__":
    main()
