# src/modules/multimodal_projection.py
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBSAVisionProjector(nn.Module):
    """
    将CLIP image encoder 输出投影到 fusion_dim。

    输入:
        x: Tensor [B, N, C_in]  (N = patch num)
    输出:
        Tensor [B, N, fusion_dim]
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # 简单的线性投影 + 层归一化
        self.proj = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        # 初始化：与 transformer 风格兼容（可选）
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C_in]
        returns: [B, N, out_dim]
        """
        if x is None:
            return None
        # 支持若输入为 [B, C, H, W] 的情形（不常见），则先 flatten 空间维度
        if x.dim() == 4:
            # assume [B, C, H, W] -> convert to [B, H*W, C]
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)

        # 常规路径 [B, N, C_in]
        out = self.proj(x)          # [B, N, out_dim]
        out = self.ln(out)
        out = self.dropout(out)
        return out


class PromptProjector(nn.Module):
    """
    将 text encoder 输出（token-level embeddings）投影到 fusion_dim。

    输入:
        x: Tensor [B, L, C_text]
    输出:
        Tensor [B, L, fusion_dim]
    注意:
        - 该模块对 token-level 特征进行 layernorm+linear 映射，保持序列结构。
        - 如果你 prefer pooling（把多个提示合并为少量 summary tokens），
          在调用本模块前先做 pooling。
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C_text]
        returns: [B, L, out_dim]
        """
        if x is None:
            return None
        # 如果输入是 2D (B, C) 代表已经 pooled 成单向向量，扩展成 seq_len=1
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, C_text]

        out = self.proj(x)
        out = self.ln(out)
        out = self.dropout(out)
        return out


# 简单的测试函数（仅在手动运行文件时执行），方便本地快速 sanity-check
if __name__ == "__main__":
    vproj = CBSAVisionProjector(in_dim=768, out_dim=512, dropout=0.1)
    pproj = PromptProjector(in_dim=512, out_dim=512, dropout=0.1)
    img = torch.randn(2, 49, 768)   # B=2, N=7x7 patches
    txt = torch.randn(2, 16, 512)   # B=2, L=16 tokens
    out_img = vproj(img)
    out_txt = pproj(txt)
    print("img ->", out_img.shape)  # (2,49,512)
    print("txt ->", out_txt.shape)  # (2,16,512)
