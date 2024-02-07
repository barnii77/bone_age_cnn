import torch.nn as nn
import torchvision.models as models


class AgePredictionCNN(nn.Module):
    """CNN model for age prediction from X-ray images."""

    @staticmethod
    def _get_new_conv_layer(
        num_in_channels, num_out_channels, kernel_size, stride, padding
    ) -> nn.Conv2d:
        """Create a new convolutional layer with the given parameters."""
        return nn.Conv2d(
            num_in_channels,
            num_out_channels,  # hardcoded because the pretrained ResNet18 model expects 64 channels
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            bias=False,
        )

    def __init__(
        self, num_channels, dropout_rate, kernel_size, stride, padding, labels_norm
    ):
        super(AgePredictionCNN, self).__init__()
        self.base_model = models.resnet18(pretrained=True)

        for param in self.base_model.parameters():
            param.requires_grad = False
        # Replace the first convolutional layer so it has the correct number of input channels
        # and make it trainable
        self.base_model.conv1 = self._get_new_conv_layer(
            num_channels, 64, kernel_size, stride, padding
        )
        for param in self.base_model.conv1.parameters():
            param.requires_grad = True
        # Unfreeze the first convolutional layer of each residual block
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(self.base_model, layer_name)
            block = layer[0]
            for param in block.conv1.parameters():
                param.requires_grad = True

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.base_model.fc.in_features, 1)
        self.base_model.fc = self.fc
        self.labels_norm = labels_norm

    def forward(self, x):
        x = self.dropout(x)
        x = self.base_model(x)
        return x
