import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 use_batch_norm=True, activation="relu"):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=not use_batch_norm),
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "leaky_relu":
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "gelu":
            layers.append(nn.GELU())

        layers.append(nn.MaxPool2d(2, 2))
        super().__init__(*layers)


class CustomCNN(nn.Module):
    def __init__(self, num_classes=100, conv_channels=None, fc_dims=None,
                 dropout=0.5, use_batch_norm=True, activation="relu"):
        super().__init__()
        if conv_channels is None:
            conv_channels = [64, 128, 256, 512]
        if fc_dims is None:
            fc_dims = [256]

        conv_blocks = []
        in_channels = 3
        for out_channels in conv_channels:
            conv_blocks.append(
                ConvBlock(in_channels, out_channels, use_batch_norm=use_batch_norm, activation=activation)
            )
            in_channels = out_channels
        self.conv_layers = nn.Sequential(*conv_blocks)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        fc_layers = []
        in_features = conv_channels[-1]
        for fc_dim in fc_dims:
            fc_layers.extend([
                nn.Linear(in_features, fc_dim),
                nn.BatchNorm1d(fc_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_features = fc_dim
        fc_layers.append(nn.Linear(in_features, num_classes))
        self.fc_layers = nn.Sequential(*fc_layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

    def get_feature_maps(self, x):
        feature_maps = []
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            feature_maps.append(x)
        return feature_maps


def build_custom_cnn(config, num_classes=100):
    return CustomCNN(
        num_classes=num_classes,
        conv_channels=config.get("conv_layers", [64, 128, 256, 512]),
        fc_dims=config.get("fc_layers", [256]),
        dropout=config.get("dropout", 0.5),
        use_batch_norm=config.get("use_batch_norm", True),
        activation=config.get("activation", "relu"),
    )
