import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import glob
import os
import pandas as pd

CSV_GLOB_PATTERN = "mediapipe_csv/*.csv"
csv_paths = sorted(glob.glob(CSV_GLOB_PATTERN))

if len(csv_paths) == 0:
    raise FileNotFoundError(f"No CSV files found matching pattern: {CSV_GLOB_PATTERN}")

for p in csv_paths:
    print(f"Found CSV file: {p}")

LABELS_DIR = "label/label"

JOINT_NAMES = [
    "NOSE",            # 0
    "LEFT_SHOULDER",   # 1
    "RIGHT_SHOULDER",  # 2
    "LEFT_ELBOW",      # 3
    "RIGHT_ELBOW",     # 4
    "LEFT_WRIST",      # 5
    "RIGHT_WRIST",     # 6
    "LEFT_HIP",        # 7
    "RIGHT_HIP",       # 8
    "LEFT_KNEE",       # 9
    "RIGHT_KNEE",      # 10
    "LEFT_ANKLE",      # 11
    "RIGHT_ANKLE",     # 12
]

pairs = [
    # Głowa -> Tułów
    ("NOSE", "LEFT_SHOULDER"),
    ("NOSE", "RIGHT_SHOULDER"),

    # Obręcz barkowa i miedniczna (poziome)
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_HIP", "RIGHT_HIP"),

    # Kręgosłup / Tułów (pionowe)
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),

    # Lewa ręka
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),

    # Prawa ręka
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),

    # Lewa noga
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),

    # Prawa noga
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
]

def build_edge_index(joint_names, connections):
    mapping = {name: i for i, name in enumerate(joint_names)}
    
    source_indices = []
    target_indices = []
    
    for src_name, tgt_name in connections:
        if src_name in mapping and tgt_name in mapping:
            src_idx = mapping[src_name]
            tgt_idx = mapping[tgt_name]
            
            source_indices.append(src_idx)
            target_indices.append(tgt_idx)
            
            source_indices.append(tgt_idx)
            target_indices.append(src_idx)
        else:
            print(f"Uwaga: Nie znaleziono stawu {src_name} lub {tgt_name} w liście!")

    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
    return edge_index

def visualize_skeleton_graph(edge_index, joint_names, specific_frame=None):
    """
    Poprawiona wersja wizualizacji dla danych znormalizowanych (MediaPipe).
    """
    num_nodes = len(joint_names)
    
    # Tworzenie grafu
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i, label=joint_names[i])
    
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)

    # Ustalanie pozycji
    pos = {}
    
    # Sprawdzamy czy mamy dane rzeczywiste
    if specific_frame is not None:
        # --- RZECZYWISTE DANE (MediaPipe) ---
        for i in range(num_nodes):
            # specific_frame to macierz [num_joints, 3] (x, y, z)
            # Uwaga: MediaPipe ma (0,0) w lewym górnym rogu.
            # Żeby wykres wyglądał naturalnie na plot (0,0 lewy dolny),
            # musimy odwrócić Y: y = -row[1]
            x = specific_frame[i][0]
            y = -specific_frame[i][1] 
            pos[i] = np.array([x, y])
            
        title = "Rzeczywista klatka (Skala 0-1)"
        offset_y = 0.04 # Małe przesunięcie dla danych 0-1
        font_size = 9
    else:
        # --- SCHEMAT T-POSE ---
        pos = {
            0: (0, 10), 1: (2, 8), 2: (-2, 8), 3: (3, 6), 4: (-3, 6),
            5: (4, 4), 6: (-4, 4), 7: (1.5, 4), 8: (-1.5, 4), 9: (1.5, 1),
            10: (-1.5, 1), 11: (1.5, -2), 12: (-1.5, -2)
        }
        # Fallback jeśli inna liczba stawów
        if len(joint_names) != 13:
            pos = nx.spring_layout(G)
            
        title = "Schemat Topologii"
        offset_y = 0.5 # Duże przesunięcie dla dużych współrzędnych
        font_size = 10

    # Kolory
    node_colors = []
    for name in joint_names:
        if "LEFT" in name: node_colors.append('#1f77b4')   # Niebieski
        elif "RIGHT" in name: node_colors.append('#d62728') # Czerwony
        else: node_colors.append('#2ca02c')                 # Zielony

    # Rysowanie
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # 1. Krawędzie
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, alpha=0.7)
    
    # 2. Węzły
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, edgecolors='black')
    
    # 3. Etykiety (z poprawionym przesunięciem)
    label_pos = {k: (v[0], v[1] - offset_y) for k, v in pos.items()} # Przesuwamy W DÓŁ
    
    # Rysujemy tekst ręcznie dla lepszej kontroli
    for k, (x, y) in label_pos.items():
        plt.text(x, y, joint_names[k], fontsize=font_size, ha='center', va='top', 
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    # Ustawienia osi dla danych rzeczywistych
    if specific_frame is not None:
        ax.set_aspect('equal') # Ważne! Żeby człowiek nie był rozciągnięty
        # Dodajemy marginesy (pobieramy zakres danych)
        all_coords = np.array(list(pos.values()))
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
        
        plt.xlim(x_min - 0.2, x_max + 0.2)
        plt.ylim(y_min - 0.2, y_max + 0.2)

    plt.title(title)
    plt.axis('off')
    plt.show()

SEQ_LEN = 32
STEP = 8

num_joints = len(JOINT_NAMES)
num_features = 2  # (x, y) positions

def load_mediapipe_with_labels(csv_paths, labels_dir, joint_names, seq_len, step):

    video_frames = []
    videos_labels_raw = []
    all_raw_labels = set()

    for coord_path in csv_paths:
        base_name = os.path.basename(coord_path)
        label_path = os.path.join(labels_dir, base_name)

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found for {coord_path}: {label_path}")
        
        df_coords = pd.read_csv(coord_path)
        grouped = df_coords.groupby('frame_number')

        df_labels = pd.read_csv(label_path, header=None)
        frame_ids = df_labels.iloc[:, 0].to_numpy().astype(int)
        class_ids = df_labels.iloc[:, 2].to_numpy().astype(int)
        frame_to_label = {f: c for f, c in zip(frame_ids, class_ids)}

        frame_nums_sorted = sorted(grouped.groups.keys())
        frames_x = []
        frames_y_raw = []

        for frame_num in frame_nums_sorted:
            frame_group = grouped.get_group(frame_num).set_index('landmark')

            frame_array = np.zeros((len(joint_names), 2), dtype=np.float32)

            for j_idx, j_name in enumerate(joint_names):
                if j_name in frame_group.index:
                    row = frame_group.loc[j_name]
                    x = float(row['x'])
                    y = float(row['y'])
                    frame_array[j_idx, 0] = x
                    frame_array[j_idx, 1] = y
                else:
                    pass

            frames_x.append(frame_array)

            frame_int = int(frame_num)
            label_raw = frame_to_label.get(frame_int, -1)
            label_raw = int(label_raw)
            if label_raw == -1:
                label_raw = 0
            
            
            frames_y_raw.append(label_raw)
            all_raw_labels.add(label_raw)

        video_frames.append(np.stack(frames_x, axis=0))
        videos_labels_raw.append(np.array(frames_y_raw, dtype=np.int64))

    unique_raw = sorted(all_raw_labels)

    label_mapping = {val: idx for idx, val in enumerate(unique_raw)}
    num_classes = len(unique_raw)

    seq_list = []
    seq_labels = []

    for X_vid, y_raw_vid in zip(video_frames, videos_labels_raw):
        y_mapped_vid = np.array([label_mapping[raw] for raw in y_raw_vid], dtype=np.int64)
        T_video = X_vid.shape[0]

        for start in range(0, T_video - seq_len + 1, step):
            end = start + seq_len
            seq_x = X_vid[start:end]
            seq_y_seq = y_mapped_vid[start:end]

            counts = np.bincount(seq_y_seq)
            seq_label = int(counts.argmax())

            seq_list.append(seq_x)
            seq_labels.append(seq_label)

        if len(seq_list) >= 100:
            break

    sequences = torch.from_numpy(np.stack(seq_list, axis=0)).float()
    labels = torch.from_numpy(np.array(seq_labels, dtype=np.int64))

    return sequences, labels, num_classes

sequences, labels, num_classes = load_mediapipe_with_labels(csv_paths, LABELS_DIR, JOINT_NAMES, SEQ_LEN, STEP)

num_sequences, T, J, F_dim = sequences.shape

edge_index = build_edge_index(JOINT_NAMES, pairs)

def sequence_to_datalist(sequence_tensor, edge_index):
    T, J, F_dim = sequence_tensor.shape

    data_list = []
    for t in range(T):
        x_t = sequence_tensor[t]
        data_t = Data(x=x_t, edge_index=edge_index)
        data_list.append(data_t)
    return data_list

all_sequences_graph = []
for i in range(num_sequences):
    data_list = sequence_to_datalist(sequences[i], edge_index)
    all_sequences_graph.append((data_list, labels[i]))

# Step 2: Create a Data object for each frame (graph data for each time step)

split_idx = int(0.8 * num_sequences)
train_seq = all_sequences_graph[:split_idx]
test_seq = all_sequences_graph[split_idx:]

print(f"Total sequences: {len(all_sequences_graph)}, Training frames: {len(train_seq)}, Testing frames: {len(test_seq)}")

# Step 3: Define the ST-GCN Model
class STGCN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(STGCN, self).__init__()
        self.gcn1 = GCNConv(in_channels, 64)
        self.gcn2 = GCNConv(64, 128)

        self.temporal_conv = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 1),
            padding=(1, 0)
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, data_list):
        spatial_features = []

        for data in data_list:
            x, edge_index = data.x, data.edge_index 
            x = F.relu(self.gcn1(x, edge_index))
            x = F.relu(self.gcn2(x, edge_index))
            spatial_features.append(x)

        x = torch.stack(spatial_features, dim=0)

        x = x.permute(2, 0, 1).unsqueeze(0)

        x = F.relu(self.temporal_conv(x))

        x = x.mean(dim=[2, 3])

        out = self.fc(x)
        return out.squeeze(0)


# Step 4: Training the ST-GCN

# Initialize the model, optimizer, and loss function
model = STGCN(in_channels=num_features, num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Function to train the model
def train(model, train_data, optimizer, criterion, epochs=200):
    model.train()
    for epoch in range(epochs):

        total_loss = 0.0

        for data, label in train_data:
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)

        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

# Step 5: Evaluation

def test(model, test_data):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in test_data:
            logits = model(data)  # Run the forward pass
            pred = logits.argmax(dim=-1)  # Get the predicted class
            if pred.item() == label.item():
                correct += 1
            total += 1

    accuracy = correct / total
    return accuracy

visualize_skeleton_graph(edge_index, JOINT_NAMES)

sample_frame = sequences[0][0].numpy()
visualize_skeleton_graph(edge_index, JOINT_NAMES, specific_frame=sample_frame)
# Train the model on the toy data
train(model, train_seq, optimizer, criterion, epochs=20)

# Test the model on the first frame (toy data)
accuracy = test(model, test_seq)
print(f'Accuracy on test data: {accuracy * 100:.2f}%')
