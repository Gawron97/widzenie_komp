import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv
from torch_geometric.utils import to_dense_adj

class STGCN(nn.Module):
    def __init__(self, in_channels, num_classes, edge_index, num_nodes, dropout=0.5):
        super(STGCN, self).__init__()
        
        self.dropout_rate = dropout
        
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        self.register_buffer('adj', adj)

        self.gcn1 = DenseGCNConv(in_channels, 64)
        self.bn1 = nn.BatchNorm1d(num_nodes)
        self.dropout1 = nn.Dropout(dropout)
        
        self.gcn2 = DenseGCNConv(64, 128)
        self.bn2 = nn.BatchNorm1d(num_nodes)
        self.dropout2 = nn.Dropout(dropout)

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.temporal_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        B, T, N, F_in = x.shape
        
        x = x.view(B * T, N, F_in)
        
        x = F.relu(self.gcn1(x, self.adj))
        x = self.bn1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.gcn2(x, self.adj))
        x = self.bn2(x)
        x = self.dropout2(x)
        
        x = x.view(B, T, N, 128)
        
        x = x.permute(0, 3, 1, 2)
        
        x = self.temporal_conv(x)
        x = self.temporal_conv2(x)
        
        x = x.mean(dim=[2, 3])
        
        x = self.fc_dropout(x)
        out = self.fc(x)
        return out