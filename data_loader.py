import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import TensorDataset

def csv_name_to_label_name(csv_basename):
    """
    Convert CSV filename to label filename.
    e.g. '001_Subject_d26arrd1.csv' -> '001_d26arrd1.csv'
    """
    # Split by '_Subject_' if present
    if '_Subject_' in csv_basename:
        parts = csv_basename.split('_Subject_')
        return parts[0] + '_' + parts[1]
    return csv_basename


def center_pose_to_hips(frame_array, hip_indices):
    """
    Przesuwa współrzędne tak, aby środek bioder był w punkcie (0,0).
    frame_array: [Joints, 2]
    """
    # Obliczamy średnią pozycję bioder (X, Y)
    hip_center = frame_array[hip_indices, :].mean(axis=0)
    # Odejmujemy środek od wszystkich stawów
    return frame_array - hip_center

def create_global_label_map(csv_paths, labels_dir):
    all_raw_labels = set()
    
    print("Building global label map...")
    for coord_path in csv_paths:
        base_name = os.path.basename(coord_path)
        label_base_name = csv_name_to_label_name(base_name)
        label_path = os.path.join(labels_dir, label_base_name)
        
        if os.path.exists(label_path):
            df_labels = pd.read_csv(label_path, header=None)
            # Kolumna 2 to ID klasy
            unique_in_file = df_labels.iloc[:, 2].unique()
            all_raw_labels.update(unique_in_file)

    # Sortujemy i tworzymy mapę {Raw_ID -> 0..N}
    unique_sorted = sorted(list(all_raw_labels))
    label_map = {val: idx for idx, val in enumerate(unique_sorted)}
    
    print(f"Found {len(label_map)} unique classes: {label_map}")
    return label_map

def load_mediapipe_with_labels(csv_paths, labels_dir, joint_names, seq_len, step, hip_indices, label_mapping):
    video_frames = []
    videos_labels_mapped = [] # Tu od razu trzymamy zmapowane labele

    for coord_path in csv_paths:
        base_name = os.path.basename(coord_path)
        label_base_name = csv_name_to_label_name(base_name)
        label_path = os.path.join(labels_dir, label_base_name)
        if not os.path.exists(label_path): continue
        
        # Wczytywanie
        df_coords = pd.read_csv(coord_path)
        grouped = df_coords.groupby('frame_number')
        
        df_labels = pd.read_csv(label_path, header=None)
        frame_to_label = {int(row[0]): int(row[2]) for _, row in df_labels.iterrows()}

        frame_nums_sorted = sorted(grouped.groups.keys())
        frames_x = []
        frames_y_mapped = []

        for frame_num in frame_nums_sorted:
            frame_group = grouped.get_group(frame_num).set_index('landmark')
            frame_array = np.zeros((len(joint_names), 2), dtype=np.float32)

            for j_idx, j_name in enumerate(joint_names):
                if j_name in frame_group.index:
                    row = frame_group.loc[j_name]
                    frame_array[j_idx] = [float(row['x']), float(row['y'])]

            # Normalizacja
            frame_array = center_pose_to_hips(frame_array, hip_indices)
            frames_x.append(frame_array)
            
            # Pobieranie i mapowanie etykiety
            label_raw = frame_to_label.get(int(frame_num), -1)
            
            # Jeśli labela nie ma w mapie (np. -1 lub błąd), dajemy 0 lub pomijamy
            # Tutaj zakładamy, że 0 to klasa "tło" lub pierwsza znaleziona
            mapped_label = label_mapping.get(label_raw, 0)
            
            frames_y_mapped.append(mapped_label)

        video_frames.append(np.stack(frames_x, axis=0))
        videos_labels_mapped.append(np.array(frames_y_mapped, dtype=np.int64))

    # Tworzenie sekwencji
    seq_list = []
    seq_labels = []

    for X_vid, y_mapped_vid in zip(video_frames, videos_labels_mapped):
        T_video = X_vid.shape[0]

        for start in range(0, T_video - seq_len + 1, step):
            end = start + seq_len
            seq_x = X_vid[start:end]
            
            # Label sekwencji to najczęstsza klasa w oknie
            counts = np.bincount(y_mapped_vid[start:end])
            seq_label = int(counts.argmax())

            seq_list.append(seq_x)
            seq_labels.append(seq_label)

    if not seq_list:
        # Zwracamy pusty dataset, jeśli pliki były za krótkie
        print("Warning: No sequences created from these files (files too short?)")
        return None

    X_tensor = torch.from_numpy(np.stack(seq_list, axis=0)).float()
    y_tensor = torch.from_numpy(np.array(seq_labels, dtype=np.int64))

    return TensorDataset(X_tensor, y_tensor)

def build_edge_index(joint_names, connections):
    mapping = {name: i for i, name in enumerate(joint_names)}
    src_idx, tgt_idx = [], []
    
    for src, tgt in connections:
        if src in mapping and tgt in mapping:
            s, t = mapping[src], mapping[tgt]
            src_idx.extend([s, t])
            tgt_idx.extend([t, s]) # Graf dwukierunkowy
    
    return torch.tensor([src_idx, tgt_idx], dtype=torch.long)
