"""
Graph Convolutional Network encoder for network topology.

Node features: [x, y, vx, vy, avail_bw_ratio, avail_comp_ratio, load]
Edge features: [sinr_norm, distance_norm, avail_bw_ratio]

Uses PyTorch Geometric (PyG) for graph operations.
H^{(l+1)} = σ(D^{-1/2} A D^{-1/2} H^{(l)} W^{(l)})
"""

from __future__ import annotations
import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GCNConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class _FallbackGCNConv(nn.Module):
    """Minimal GCN layer when PyG is not installed."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.linear = nn.Linear(in_c, out_c)

    def forward(self, x, edge_index):
        # Simple message passing: aggregate neighbour features
        n = x.size(0)
        row, col = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col])
        deg = torch.zeros(n, device=x.device)
        deg.index_add_(0, row, torch.ones(row.size(0), device=x.device))
        deg = deg.clamp(min=1).unsqueeze(1)
        agg = agg / deg
        return self.linear(agg + x)


class GCNEncoder(nn.Module):
    """2-layer GCN that produces per-node embeddings."""

    def __init__(self, in_features: int = 7, hidden: int = 64,
                 out_features: int = 32):
        super().__init__()
        Conv = GCNConv if HAS_PYG else _FallbackGCNConv
        self.conv1 = Conv(in_features, hidden)
        self.conv2 = Conv(hidden, out_features)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_features)
        edge_index: (2, E)
        returns: (N, out_features)
        """
        h = self.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h
