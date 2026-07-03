import os
import glob
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


class LRSchedulerCallback:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def step(self, metrics=None):
        if metrics is not None:
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()


class Trainer:
    def __init__(self, model, criterion=None, optimizer=None, scheduler=None,
                 device=None, save_dir="saved_models", use_wandb=False):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "lr": [], "epoch_time": [],
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0 * correct / total:.2f}%"
            })

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader, epochs=50,
              early_stopping=None, model_name="model"):
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            epoch_start = time.time()

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)
            self.history["epoch_time"].append(epoch_time)

            if self.scheduler:
                self.scheduler.step()

            if self.use_wandb:
                import wandb
                wandb.log({
                    "train_loss": train_loss, "val_loss": val_loss,
                    "train_acc": train_acc, "val_acc": val_acc,
                    "learning_rate": current_lr, "epoch_time": epoch_time,
                })

            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.save_checkpoint(model_name, epoch, val_acc)

            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
                  f"{'  *BEST*' if is_best else ''}")

            if early_stopping and early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        print(f"\nBest Val Accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch+1}")
        return self.history

    def save_checkpoint(self, model_name, epoch, val_acc):
        path = self.save_dir / f"{model_name}_best.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
        }, path)

    def load_checkpoint(self, model_name):
        path = self.save_dir / f"{model_name}_best.pth"
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            return checkpoint
        return None

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for inputs, targets in test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        inference_time = time.time() - start_time
        test_loss = running_loss / total
        test_acc = 100.0 * correct / total

        return {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "predictions": np.array(all_preds),
            "targets": np.array(all_targets),
            "probabilities": np.array(all_probs),
            "inference_time": inference_time,
            "time_per_image": inference_time / total * 1000,
        }

    def get_model_size(self):
        path = self.save_dir / f"*_best.pth"
        files = glob.glob(str(path))
        if files:
            size_mb = sum(os.path.getsize(f) for f in files) / (1024 * 1024)
            return size_mb
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
