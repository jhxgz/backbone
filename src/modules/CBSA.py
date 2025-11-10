from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

"""
收缩-广播自注意力模块（Contract-and-Broadcast Self-Attention, CBSA）
1、核心机理：智能提炼，高效传播
    CSBA模块的核心在于压缩与广播的智能机制。他不像传统注意力那样平等处理所有特征，而是像一位经验丰富的编辑，能从海量信息中精准抓取少数代表性关键特征
    进行深度提炼与加工，再将提炼后的精华高效地广播给所有特征。这确保了模型能够聚焦于最重要的信息，从而提升了决策质量。
2、核心价值：兼顾性能与效率
    这个模块为模型带来了鱼和熊掌兼得的效果。一方面，通过聚焦和强化关键特征，它显著提升了模型的表征能力和最终性能。另一方面，其独特的处理方式将计算复杂
    度从二次降为线性，使得模型在保持高性能的同时，运行更快、消耗资源更少，尤其适合处理高分辨率、长序列等复杂数据。
"""

class CBSA(nn.Module):
    """
    Contract-and-Broadcast Self-Attention (简化稳妥版)
    - 支持 CLS 位于开头/末尾/无 CLS
    - 自适应从 N 推出 HxW 网格（若 (N-1) 可开方 → 认为有 CLS；否则尝试 N 可开方）
    - 代表网格大小可配置 (rep_h, rep_w)
    - 输出 = x + Δx  （残差更稳）
    - 接口: (B, N, D) -> (B, N, D)
    """
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: Optional[int] = None,
        rep_h: int = 8,
        rep_w: int = 8,
        cls_pos: str = "first",   # "first" | "last" | "none"
        dropout: float = 0.0,
    ):
        super().__init__()
        assert heads > 0
        if dim_head is None:
            assert dim % heads == 0, "dim must be divisible by heads"
            dim_head = dim // heads

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.rep_h = rep_h
        self.rep_w = rep_w
        self.cls_pos = cls_pos

        inner_dim = heads * dim_head
        self.to_inner = nn.Linear(dim, inner_dim, bias=False)
        self.to_out   = nn.Linear(inner_dim, dim, bias=False)

        # 可学习步长（收缩量、广播量）
        self.step_x   = nn.Parameter(torch.ones(heads, 1, 1))
        self.step_rep = nn.Parameter(torch.ones(heads, 1, 1))

        # 自适应池化得到代表（默认从 token 网格池化到 rep_h x rep_w）
        self.pool = nn.AdaptiveAvgPool2d(output_size=(rep_h, rep_w))

        self.attend = nn.Softmax(dim=-1)
        self.drop   = nn.Dropout(dropout)

    @staticmethod
    def _infer_grid(n_tokens_no_cls: int) -> int:
        """从 token 数量推出网格边长（平方数情况），失败则返回 -1。"""
        s = int((n_tokens_no_cls) ** 0.5)
        return s if s * s == n_tokens_no_cls else -1

    def _split_cls(self, x: torch.Tensor):
        """
        根据 cls_pos 分出 cls 与 patch：返回 (cls_token or None, patches)
        x: [B, H, N, Hd]
        """
        if self.cls_pos == "first" and x.size(2) >= 2:   # 注意：这里用的是 dim=2 (token 维)
            return x[:, :, :1, :], x[:, :, 1:, :]
        if self.cls_pos == "last" and x.size(2) >= 2:    # 同样用 dim=2
            return x[:, :, -1:, :], x[:, :, :-1, :]
        # none
        return None, x


    def attention(self, q, k, v):
        # 通用缩放点积注意力
        dots = (q @ k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out  = attn @ v
        return out, attn

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        x: [B, N, D]
        return: [B, N, D] 或注意力（可视化）
        """
        B, N, D = x.shape
        inner = self.to_inner(x)                     # [B, N, H*Hd]
        inner = rearrange(inner, 'b n (h d) -> b h n d', h=self.heads, d=self.dim_head)  # [B,H,N,Hd]

        # 划分 cls 与 patch
        cls_tok, patches = self._split_cls(inner)    # cls_tok:[B,H,1,Hd] or None; patches:[B,H,NP,Hd]
        NP = patches.size(2)

        # 推出网格
        side = self._infer_grid(NP)
        if side < 0:
            # 兜底：把序列当做 1 x NP 网格进行池化（可跑但不如方阵稳）
            H_grid, W_grid = 1, NP
        else:
            H_grid = W_grid = side

        # --- 把 [B,H,NP,Hd] 整成 2D 网格后做池化 ---

        if side < 0:
            # 兜底：当 NP 不是平方数时，就按 1×NP 的“长条网格”
            # 先转成 [B,H,Hd,1,NP]，然后合并 B 和 H → [B*H,Hd,1,NP]
            patches_grid = rearrange(patches, 'b h n d -> (b h) d 1 n')
        else:
            # 正常方阵： [B,H,NP,Hd] → [B*H, Hd, H_grid, W_grid]
            patches_grid = rearrange(patches, 'b h (p1 p2) d -> (b h) d p1 p2', p1=H_grid, p2=W_grid)

        # 现在是 4D：[B*H, Hd, H_grid, W_grid]，可以喂给 adaptive_avg_pool2d
        reps = self.pool(patches_grid)  # [B*H, Hd, Rh, Rw]

        # 还原回 [B,H,R,Hd]
        reps = rearrange(reps, '(b h) d rh rw -> b h (rh rw) d', h=self.heads)  # [B,H,R,Hd]

        R = reps.size(2)

        # 注意：让 reps 去 attention 到 patches（Q=rep,K=patch,V=patch）
        rep_delta, attn = self.attention(reps, patches, patches)    # rep_delta:[B,H,R,Hd]  attn:[B,H,R,NP]

        if return_attn:
            # 返回 token←rep 的“影响力”矩阵（大概可视化）
            # [B,H,NP,NP] ≈ A^T @ A
            return attn.transpose(-1, -2) @ attn

        reps = reps + self.step_rep * rep_delta                     # 收缩代表

        # 广播：用代表之间的自注意促成“语义一致”的扩散，再把代表通过 A^T 扩回 tokens
        rep_to_rep, _ = self.attention(reps, reps, reps)            # [B,H,R,Hd]
        delta_tokens   = attn.transpose(-1, -2) @ rep_to_rep        # [B,H,NP,Hd]
        delta_tokens   = self.step_x * delta_tokens                 # [B,H,NP,Hd]

        # 拼回 cls，并还原到 [B,N,D]
        if cls_tok is not None:
            tokens = torch.cat([cls_tok, delta_tokens], dim=2) if self.cls_pos == "first" else torch.cat([delta_tokens, cls_tok], dim=2)
        else:
            tokens = delta_tokens                                   # [B,H,N,Hd]

        delta = rearrange(tokens, 'b h n d -> b n (h d)')           # [B,N,H*Hd]
        y = x + self.drop(self.to_out(delta))                        # 残差输出
        return y
