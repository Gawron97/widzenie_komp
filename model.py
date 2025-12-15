import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv
from torch_geometric.utils import to_dense_adj

class STGCN(nn.Module):
    def __init__(self, in_channels, num_classes, edge_index, num_nodes):
        super(STGCN, self).__init__()
        
        # Konwersja edge_index na macierz sąsiedztwa (Dense Adjacency Matrix)
        # Wynik: [num_nodes, num_nodes]
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        # Rejestrujemy jako bufor (nie parametr do uczenia, ale stała część modelu)
        self.register_buffer('adj', adj)

        # 1. Spatial Blocks (Dense GCN)
        # DenseGCNConv akceptuje wejście (Batch, Nodes, Features)
        self.gcn1 = DenseGCNConv(in_channels, 64)
        self.bn1 = nn.BatchNorm1d(num_nodes) # Normalizacja po węzłach
        
        self.gcn2 = DenseGCNConv(64, 128)
        self.bn2 = nn.BatchNorm1d(num_nodes)

        # 2. Temporal Block
        # Zwiększamy kernel (widzi 9 klatek naraz) i dodajemy padding
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5) # <-- NOWOŚĆ: Dropout (zapobiega overfit)
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Wejście x: (Batch, Time, Nodes, Features)
        B, T, N, F_in = x.shape
        
        # --- SPATIAL (GCN) ---
        # Musimy przetworzyć każdą klatkę. DenseGCNConv obsługuje Batch.
        # Spłaszczamy czas do wymiaru batcha: (Batch * Time, Nodes, Features)
        x = x.view(B * T, N, F_in)
        
        # GCN 1
        x = F.relu(self.gcn1(x, self.adj)) # self.adj jest stałe dla wszystkich
        x = self.bn1(x) # BatchNorm
        
        # GCN 2
        x = F.relu(self.gcn2(x, self.adj))
        x = self.bn2(x)
        
        # Przywracamy wymiary: (Batch, Time, Nodes, Channels)
        # Wynik GCN ma 128 kanałów
        x = x.view(B, T, N, 128)
        
        # --- TEMPORAL (Conv2d) ---
        # Permutacja pod Conv2d: (Batch, Channels, Time, Nodes)
        x = x.permute(0, 3, 1, 2)
        
        x = self.temporal_conv(x)
        
        # --- GLOBAL POOLING ---
        x = x.mean(dim=[2, 3]) # Średnia po Czasie i Węzłach
        
        # --- CLASSIFICATION ---
        out = self.fc(x)
        return out