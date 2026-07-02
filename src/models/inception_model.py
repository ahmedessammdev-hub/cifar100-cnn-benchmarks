import torch
import torch.nn as nn
from torchvision import models


class InceptionV3Transfer(nn.Module):
    def __init__(self, num_classes=100, mode="feature_extraction", fine_tune_layers=3):
        super().__init__()
        self.mode = mode
        self.base_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)

        if mode == "feature_extraction":
            for param in self.base_model.parameters():
                param.requires_grad = False
        elif mode == "fine_tuning":
            for param in self.base_model.parameters():
                param.requires_grad = False
            children = list(self.base_model.children())
            for child in children[-fine_tune_layers:]:
                for param in child.parameters():
                    param.requires_grad = True

        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)
        self.base_model.aux_logits = False

        self.input_size = 299

    def forward(self, x):
        return self.base_model(x)

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def get_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
