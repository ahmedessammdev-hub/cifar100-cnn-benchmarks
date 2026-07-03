import torch
import torch.optim as optim


def get_optimizer(model, config):
    lr = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 0.0001)

    return optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )


def get_scheduler(optimizer, config):
    scheduler_type = config.get("scheduler", "cosine")
    epochs = config.get("epochs", 50)

    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    elif scheduler_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )
    elif scheduler_type == "warmup_cosine":
        warmup_epochs = config.get("warmup_epochs", 5)
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs
        )
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        return optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
        )
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def get_transfer_optimizer(model, mode="feature_extraction", lr=0.001):
    if mode == "feature_extraction":
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        return optim.Adam(trainable_params, lr=lr, weight_decay=0.0001)
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        return optim.Adam(trainable_params, lr=lr, weight_decay=0.0001)
