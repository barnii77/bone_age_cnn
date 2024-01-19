import torch.nn as nn
import torchvision.models as models


class AgePredictionCNN(nn.Module):
    """CNN model for age prediction from X-ray images."""

    def __init__(
        self, num_channels, dropout_rate, kernel_size, stride, padding, labels_norm
    ):
        super(AgePredictionCNN, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.conv1 = nn.Conv2d(
            num_channels,
            64,  # hardcoded because the pretrained ResNet18 model expects 64 channels
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            bias=False,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.base_model.fc.in_features, 1)
        self.base_model.fc = self.fc
        self.labels_norm = labels_norm

    def forward(self, x):
        x = self.base_model(x)
        return x
