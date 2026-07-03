import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, target_class, output.detach()

    def overlay_cam(self, image, cam, alpha=0.5):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = np.transpose(image, (1, 2, 0))
            image = (image - image.min()) / (image.max() - image.min())
            image = (image * 255).astype(np.uint8)

        if isinstance(cam, torch.Tensor):
            cam = cam.cpu().numpy()

        cam = cam.squeeze()
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
        cam = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        overlay = (1 - alpha) * image + alpha * heatmap
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return overlay


def get_last_conv_layer(model):
    import torch.nn as nn
    conv_layers = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(module)
    if conv_layers:
        return conv_layers[-1]
    raise ValueError("No Conv2d layer found in the model")


def get_target_layer_for_model(model):
    import torchvision.models as models

    model_type = type(model).__name__

    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "features"):
            for layer in reversed(list(base.features)):
                if hasattr(layer, "weight"):
                    return layer
        if hasattr(base, "layer4"):
            return base.layer4[-1]
        if hasattr(base, "features"):
            return list(base.features)[-1]

    for module in reversed(list(model.modules())):
        import torch.nn as nn
        if isinstance(module, nn.Conv2d):
            return module

    raise ValueError("Could not find target layer")
