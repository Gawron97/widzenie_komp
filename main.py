import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from visualizer import visualize_skeleton_human

import config as cfg
import data_loader as dl
import matplotlib.pyplot as plt
from model import STGCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def main():
    print(f"--- START on {device} ---")
    
    # 1. Pobranie wszystkich ścieżek
    try:
        all_csv_paths = cfg.get_csv_paths()
    except Exception as e:
        print(e)
        return

    # 2. Stworzenie GLOBALNEJ mapy etykiet (na podstawie wszystkich plików)
    # To kluczowe, żeby ID klasy 1 w Train znaczyło to samo co ID 1 w Test
    global_label_map = dl.create_global_label_map(all_csv_paths, cfg.LABELS_DIR)
    num_classes = len(global_label_map)

    # 3. Podział plików na Train i Test (Split by File)
    random.seed(42) # Dla powtarzalności
    random.shuffle(all_csv_paths)
    
    split_ratio = 0.8
    split_idx = int(len(all_csv_paths) * split_ratio)
    
    train_paths = all_csv_paths[:split_idx]
    test_paths = all_csv_paths[split_idx:]
    
    print(f"Total files: {len(all_csv_paths)}")
    print(f"Train files: {len(train_paths)}")
    print(f"Test files:  {len(test_paths)}")

    # 4. Ładowanie danych osobno dla Train i Test
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

    # 5. DataLoaders
    use_pin_memory = (device.type == 'cuda')
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, pin_memory=use_pin_memory)
    # Testowego loadera nie musimy shuffle'ować
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, pin_memory=use_pin_memory)

    print(f"Train sequences: {len(train_dataset)}, Test sequences: {len(test_dataset)}")

    # 6. Model
    edge_index = dl.build_edge_index(cfg.JOINT_NAMES, cfg.SKELETON_PAIRS).to(device)
    model = STGCN(
        in_channels=cfg.NUM_FEATURES, 
        num_classes=num_classes,
        edge_index=edge_index,
        num_nodes=len(cfg.JOINT_NAMES)
    ).to(device)

    edge_index_vis = edge_index.detach().cpu()
    visualize_skeleton_human(edge_index_vis, cfg.JOINT_NAMES)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    # 7. Trening

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    epochs = 30
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | "
                  f"Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.1f}%")
        else:
            print(f"Epoch {epoch+1:02d}/{epochs} | "
                  f"Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
                  f"Test Loss: {test_loss:.4f}")
            
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs+1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Test Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()