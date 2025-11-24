import torch
import torch.nn as nn
import torch.nn.functional as F

class TextAdapterResidual(nn.Module):
    """
    Text-side Adapter Residual。
    可用于 (B, T, C) 或 (B, C) 两类输入。
    使用方式：
        ta = TextAdapterResidual(dim=C, down_ratio=8, use_gate=True, init_gate=-2.0)
        out = ta(text_feats)  # shape preserved
    """
    def __init__(self, dim, down_ratio=8, dropout=0.1, use_gate=True, init_gate=-2.0):
        super().__init__()
        mid = max(8, dim // down_ratio)
        self.ln = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, mid)
        self.act = nn.GELU()
        self.up = nn.Linear(mid, dim)
        self.dropout = nn.Dropout(dropout)
        self.use_gate = use_gate
        if use_gate:
            # per-channel gate
            self.gate = nn.Parameter(torch.ones(dim) * init_gate)
        else:
            self.gate = None

        # 初始化
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.constant_(self.down.bias, 0.)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.constant_(self.up.bias, 0.)

    def forward(self, x):
        """
        x: (B, T, C) or (B, C)
        returns same shape
        """
        orig_shape = x.shape
        if x.dim() == 2:
            # (B, C) -> make (B, 1, C) to reuse code
            x_in = x.unsqueeze(1)
            squeeze_out = True
        else:
            x_in = x
            squeeze_out = False

        # LN + bottleneck MLP applied token-wise
        y = self.ln(x_in)
        y = self.down(y)
        y = self.act(y)
        y = self.up(y)
        y = self.dropout(y)

        if self.use_gate:
            g = torch.sigmoid(self.gate)  # (C,)
            y = y * g.view(1, 1, -1)

        out = x_in + y  # residual
        if squeeze_out:
            return out.squeeze(1)
        return out
