import torch
import torch.nn as nn
from torchvision import models


class VGG16Transfer(nn.Module):
    def __init__(self, num_classes=100, mode="feature_extraction", fine_tune_layers=3):
        super().__init__()
        self.mode = mode
        self.base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        if mode == "feature_extraction":
            for param in self.base_model.parameters():
                param.requires_grad = False
        elif mode == "fine_tuning":
            for param in self.base_model.parameters():
                param.requires_grad = False
            total_layers = len(list(self.base_model.features))
            for i, param in enumerate(self.base_model.features.parameters()):
                if i >= total_layers - fine_tune_layers:
                    param.requires_grad = True

        in_features = self.base_model.classifier[-1].in_features
        self.base_model.classifier[-1] = nn.Linear(in_features, num_classes)

        if mode == "feature_extraction":
            for param in self.base_model.classifier.parameters():
                param.requires_grad = True

        self.input_size = 224

    def forward(self, x):
        return self.base_model(x)

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def get_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
