import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import argparse
import os
import numpy as np
from visualizer import visualize_skeleton_human

import config as cfg
import data_loader as dl
import matplotlib.pyplot as plt
from model import STGCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model save path
MODEL_PATH = "stgcn_model.pth"


class DataAugmentation:
    """Data augmentation for skeleton sequences."""
    
    def __init__(self, noise_std=0.01, time_shift_max=3, scale_range=(0.9, 1.1)):
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max
        self.scale_range = scale_range
    
    def add_noise(self, x):
        """Add Gaussian noise to coordinates."""
        noise = torch.randn_like(x) * self.noise_std
        return x + noise
    
    def random_scale(self, x):
        """Random scaling of coordinates."""
        scale = random.uniform(*self.scale_range)
        return x * scale
    
    def time_shift(self, x):
        """Random temporal shift (circular)."""
        shift = random.randint(-self.time_shift_max, self.time_shift_max)
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=0)
        return x
    
    def __call__(self, x):
        """Apply all augmentations with probability."""
        if random.random() < 0.5:
            x = self.add_noise(x)
        if random.random() < 0.3:
            x = self.random_scale(x)
        if random.random() < 0.3:
            x = self.time_shift(x)
        return x


class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies augmentation during training."""
    
    def __init__(self, dataset, augmentation=None, training=True):
        self.dataset = dataset
        self.augmentation = augmentation
        self.training = training
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.training and self.augmentation is not None:
            x = self.augmentation(x)
        return x, y


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=20, min_delta=0.001, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = None
        self.best_acc = None
        self.best_model_state = None
        self.early_stop = False
    
    def __call__(self, val_loss, val_acc, model):
        if self.best_acc is None or val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def restore_best_weights(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print(f"Restored best model weights (Test Acc: {self.best_acc*100:.1f}%)")


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)
        
    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def save_model(model, optimizer, epoch, num_classes, best_acc, path=MODEL_PATH):
    """Save model checkpoint with training state."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_classes': num_classes,
        'best_acc': best_acc,
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path} (Epoch {epoch}, Acc: {best_acc*100:.1f}%)")


def load_model(model, optimizer=None, path=MODEL_PATH):
    """Load model checkpoint. Returns epoch number and best accuracy."""
    if not os.path.exists(path):
        print(f"No saved model found at {path}")
        return None, None
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0)
    print(f"Model loaded from {path} (Epoch {epoch}, Best Acc: {best_acc*100:.1f}%)")
    return epoch, best_acc


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ST-GCN Action Recognition Training with Anti-Overfitting')
    parser.add_argument('--load', action='store_true', 
                        help='Load pre-trained model instead of training')
    parser.add_argument('--epochs', type=int, default=300, 
                        help='Maximum training epochs (default: 300)')
    parser.add_argument('--model-path', type=str, default=MODEL_PATH,
                        help=f'Path to save/load model (default: {MODEL_PATH})')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience (default: 30 epochs)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='L2 regularization weight decay (default: 1e-4)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    args = parser.parse_args()
    
    print(f"=" * 60)
    print(f"ST-GCN Training with Anti-Overfitting Techniques")
    print(f"=" * 60)
    print(f"Device: {device}")
    print(f"Mode: {'LOAD' if args.load else 'TRAIN'}")
    print(f"Max Epochs: {args.epochs}")
    print(f"Early Stopping Patience: {args.patience}")
    print(f"Dropout: {args.dropout}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Initial LR: {args.lr}")
    print(f"Data Augmentation: {'OFF' if args.no_augment else 'ON'}")
    print(f"=" * 60)
    
    try:
        all_csv_paths = cfg.get_csv_paths()
    except Exception as e:
        print(e)
        return

    global_label_map = dl.create_global_label_map(all_csv_paths, cfg.LABELS_DIR)
    num_classes = len(global_label_map)

    random.seed(42)
    random.shuffle(all_csv_paths)
    
    split_ratio = 0.8
    split_idx = int(len(all_csv_paths) * split_ratio)
    
    train_paths = all_csv_paths[:split_idx]
    test_paths = all_csv_paths[split_idx:]
    
    print(f"Total files: {len(all_csv_paths)}")
    print(f"Train files: {len(train_paths)}")
    print(f"Test files:  {len(test_paths)}")

    print("Processing TRAIN data...")
    train_dataset = dl.load_mediapipe_with_labels(
        train_paths, cfg.LABELS_DIR, cfg.JOINT_NAMES, 
        cfg.SEQ_LEN, cfg.STEP, cfg.HIP_INDICES, global_label_map
    )
    
    print("Processing TEST data...")
    test_dataset = dl.load_mediapipe_with_labels(
        test_paths, cfg.LABELS_DIR, cfg.JOINT_NAMES, 
        cfg.SEQ_LEN, cfg.STEP, cfg.HIP_INDICES, global_label_map
    )

    if train_dataset is None or test_dataset is None:
        print("Error: Not enough data created.")
        return

    # 5. Apply data augmentation to training set
    augmentation = None if args.no_augment else DataAugmentation(
        noise_std=0.01,
        time_shift_max=2,
        scale_range=(0.95, 1.05)
    )
    
    train_dataset_aug = AugmentedDataset(train_dataset, augmentation, training=True)
    test_dataset_wrapped = AugmentedDataset(test_dataset, None, training=False)

    # 6. DataLoaders
    use_pin_memory = (device.type == 'cuda')
    num_workers = 0
    
    train_loader = DataLoader(
        train_dataset_aug, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        pin_memory=use_pin_memory,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset_wrapped, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False, 
        pin_memory=use_pin_memory,
        num_workers=num_workers
    )

    print(f"Train sequences: {len(train_dataset)}, Test sequences: {len(test_dataset)}")

    # 7. Model with configurable dropout
    edge_index = dl.build_edge_index(cfg.JOINT_NAMES, cfg.SKELETON_PAIRS).to(device)
    model = STGCN(
        in_channels=cfg.NUM_FEATURES, 
        num_classes=num_classes,
        edge_index=edge_index,
        num_nodes=len(cfg.JOINT_NAMES),
        dropout=args.dropout
    ).to(device)
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler - reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=True
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    # 8. Load or Train
    if args.load:
        loaded_epoch, loaded_acc = load_model(model, path=args.model_path)
        if loaded_epoch is None:
            print("Cannot load model. Run without --load to train a new model.")
            return
        
        # Evaluate the loaded model
        print("\nEvaluating loaded model...")
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.1f}%")
        
    else:
        # Train new model with anti-overfitting
        epochs = args.epochs
        print(f"\nStarting training for up to {epochs} epochs...")
        print(f"Early stopping will trigger after {args.patience} epochs without improvement.\n")
        
        # Early stopping
        early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)
        
        train_losses, test_losses = [], []
        train_accs, test_accs = [], []
        lrs = []
        best_test_acc = 0.0

        for epoch in range(epochs):
            current_lr = optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
            test_loss, test_acc = evaluate(model, test_loader, criterion)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            # Update learning rate based on test accuracy
            scheduler.step(test_acc)
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                save_model(model, optimizer, epoch + 1, num_classes, best_test_acc, args.model_path)
            
            # Check early stopping
            if early_stopping(test_loss, test_acc, model):
                print(f"\n{'='*60}")
                print(f"Early stopping triggered at epoch {epoch + 1}")
                print(f"Best Test Accuracy: {early_stopping.best_acc*100:.1f}%")
                print(f"{'='*60}\n")
                early_stopping.restore_best_weights(model)
                epochs = epoch + 1
                break
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:03d}/{args.epochs} | "
                      f"Loss: {train_loss:.4f} | Train: {train_acc*100:.1f}% | "
                      f"Test: {test_acc*100:.1f}% | LR: {current_lr:.2e} | "
                      f"ES: {early_stopping.counter}/{early_stopping.patience}")
            else:
                print(f"Epoch {epoch+1:03d}/{args.epochs} | "
                      f"Loss: {train_loss:.4f} | Train: {train_acc*100:.1f}% | "
                      f"Test Loss: {test_loss:.4f}")
        
        # Final evaluation
        final_test_loss, final_test_acc = evaluate(model, test_loader, criterion)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss plot
        axes[0, 0].plot(range(1, len(train_losses)+1), train_losses, label="Train Loss", alpha=0.8)
        axes[0, 0].plot(range(1, len(test_losses)+1), test_losses, label="Test Loss", alpha=0.8)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Train vs Test Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(range(1, len(train_accs)+1), [a*100 for a in train_accs], label="Train Acc", alpha=0.8)
        axes[0, 1].plot(range(1, len(test_accs)+1), [a*100 for a in test_accs], label="Test Acc", alpha=0.8)
        axes[0, 1].axhline(y=best_test_acc*100, color='r', linestyle='--', label=f'Best: {best_test_acc*100:.1f}%')
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy (%)")
        axes[0, 1].set_title("Train vs Test Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 0].plot(range(1, len(lrs)+1), lrs, label="Learning Rate", color='green')
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Learning Rate")
        axes[1, 0].set_title("Learning Rate Schedule")
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Overfitting gap plot
        gap = [t*100 - v*100 for t, v in zip(train_accs, test_accs)]
        axes[1, 1].plot(range(1, len(gap)+1), gap, label="Train-Test Gap", color='purple')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Accuracy Gap (%)")
        axes[1, 1].set_title("Overfitting Gap (Train Acc - Test Acc)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150)
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Final Test Accuracy: {final_test_acc*100:.1f}%")
        print(f"Best Test Accuracy:  {best_test_acc*100:.1f}%")
        print(f"Epochs trained: {len(train_losses)}")
        print(f"Model saved to: {args.model_path}")
        print(f"Training curves saved to: training_curves.png")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()