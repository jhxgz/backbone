import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FiLMTokenizerFusion(nn.Module):
    """
    FiLM 融合模块（针对 token 形式的 visual features）。
    设计目标：用 text_features 生成 gamma/beta，对 image tokens 做 feature-wise linear modulation。
    输入：
      - image_tokens: (B, N, C)
      - text_feats:   (B, C) 或 (B, T, C)
    输出：
      - (B, N, C) （与输入 shape 相同）
    主要参数：
      - dim: 图像 token 的通道数 C（CLIP 的 hidden dim）
      - text_dim: 文本特征维度（如果 None 则认为等于 dim 或会做映射）
      - hidden: MLP 隐层维度
      - per_token: 是否对每个 token 生成独立 gamma/beta（False：全局通道级广播；True：(B,N,C)）
      - use_residual: 是否使用 residual 输出 x + alpha * film(...)
      - init_alpha: alpha 初始值（小值更稳，推荐 0.1 或 0.05）
    """
    def __init__(self,
                 dim: int,
                 text_dim: Optional[int] = None,
                 hidden: Optional[int] = None,
                 per_token: bool = False,
                 use_residual: bool = True,
                 init_alpha: float = 0.1,
                 text_pool: str = "mean",
                 dropout: float = 0.0):
        super().__init__()
        assert text_pool in ("mean", "cls"), "text_pool must be 'mean' or 'cls'"
        self.dim = dim
        self.text_dim = text_dim or dim
        self.hidden = hidden or max(dim, self.text_dim)
        self.per_token = per_token
        self.use_residual = use_residual
        self.text_pool = text_pool

        # 如果 text_dim != dim，就把 text 映射到 dim
        if self.text_dim != self.dim:
            self.text_proj = nn.Linear(self.text_dim, self.dim)
        else:
            self.text_proj = None

        # MLP 生成 gamma 和 beta（通道级）
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, self.dim * 2)  # -> [gamma, beta]
        )

        # 若 per_token=True，使用一个轻量的 token_projection 将（B,C）扩展为 token-aware 更新
        if self.per_token:
            # 用简洁的 two-layer proj 做 token-aware 转换
            self.token_proj = nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.GELU(),
                nn.Linear(self.dim, self.dim)
            )
        else:
            self.token_proj = None

        # layernorm 在 image tokens 上（可选但推荐）
        self.image_ln = nn.LayerNorm(dim)

        # residual scale alpha（可训练）
        if self.use_residual:
            self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        else:
            self.alpha = None

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _pool_text(self, text_feats: torch.Tensor):
        """
        支持 text_feats: (B, C) 或 (B, T, C)
        返回 pooled_text: (B, C)
        """
        if text_feats.dim() == 2:
            return text_feats
        elif text_feats.dim() == 3:
            if self.text_pool == "mean":
                return text_feats.mean(dim=1)
            else:
                # 若有 explicit CLS token，建议传入 CLS，fallback 使用第0个token
                return text_feats[:, 0, :]
        else:
            raise ValueError("text_feats should be (B,C) or (B,T,C)")

    def forward(self, image_tokens: torch.Tensor, text_feats: torch.Tensor):
        """
        image_tokens: (B, N, C)
        text_feats: (B, C) or (B, T, C)
        returns: (B, N, C)
        """
        B, N, C = image_tokens.shape
        assert C == self.dim, f"image token dim {C} != model dim {self.dim}"

        # 1) 可选的 layer norm
        x_ln = self.image_ln(image_tokens)  # (B,N,C)

        # 2) pool text -> (B, text_dim)
        pooled = self._pool_text(text_feats)  # (B, text_dim)
        if self.text_proj is not None:
            pooled = self.text_proj(pooled)  # (B, dim)

        # 3) MLP 生成 gamma, beta （通道级）
        film_params = self.mlp(pooled)  # (B, 2*dim)
        gamma, beta = film_params.chunk(2, dim=-1)  # each (B, dim)

        # 让 gamma 初始靠近 1，beta 靠近 0（更稳）
        gamma = 1.0 + 0.01 * gamma
        beta = 0.01 * beta

        if self.per_token:
            # token-aware 扩展
            # 先把 (B, dim) -> (B, dim) via token_proj，然后 broadcast 到 tokens
            token_gain = self.token_proj(gamma)  # (B, dim)
            token_bias = self.token_proj(beta)   # (B, dim)
            token_gain = token_gain.unsqueeze(1).expand(B, N, C)
            token_bias = token_bias.unsqueeze(1).expand(B, N, C)
            out = token_gain * x_ln + token_bias
        else:
            # 广播到 token 轴
            gamma = gamma.unsqueeze(1).expand(B, N, C)
            beta = beta.unsqueeze(1).expand(B, N, C)
            out = gamma * x_ln + beta  # (B,N,C)

        if self.use_residual:
            return image_tokens + self.alpha * out
        else:
            return out
