import torch
import torch.nn as nn
import torch.nn.functional as F

class AdapterResidual(nn.Module):
    def __init__(self, dim, down_ratio=4, dropout=0.1, use_gate=True):
        super().__init__()
        mid = dim // down_ratio
        self.ln = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, mid)
        self.act = nn.GELU()
        self.up = nn.Linear(mid, dim)
        self.dropout = nn.Dropout(dropout)
        self.use_gate = use_gate
        if use_gate:
            # scalar gate per channel (or one scalar). Here per-channel
            self.gate = nn.Parameter(torch.ones(dim) * -2.0)  # sigmoid(-2) ~ 0.12 initial
        else:
            self.gate = None

    def forward(self, x):
        # x: (B, N, D)
        y = self.ln(x)
        y = self.down(y)    # (B,N,mid)
        y = self.act(y)
        y = self.up(y)      # (B,N,D)
        y = self.dropout(y)
        if self.use_gate:
            g = torch.sigmoid(self.gate)  # (D,)
            y = y * g.view(1,1,-1)
        return y  # caller should do x + y
