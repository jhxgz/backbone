import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """
    [修改版] 动态门控融合模块
    新增功能：支持外部传入 reliability (置信度)，实现显式去噪。
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim

        # 1. 内部门控网络 (保持不变)
        # 计算 "Feature-based Gate": 基于当前的数值特征判断融合比例
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        # 2. 特征变换层
        self.visual_transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU()
        )
        self.text_transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU()
        )

        self.ln = nn.LayerNorm(dim)

    def forward(self, img_feats, text_feats, reliability=None):
        """
        Args:
            img_feats: (B, N, C)
            text_feats: (B, N, C) - 对齐后的文本特征
            reliability: (B, 1, C) or (B, N, C) - [新增] 来自 DPG 的去噪信号
                         如果为 None，则退化为普通 GatedFusion
        """
        # 1. 特征预处理
        v_prime = self.visual_transform(img_feats)
        t_prime = self.text_transform(text_feats)

        # 2. 计算内部门控 (Internal Gate)
        # 这里的 gate 只是基于“长得像不像”来判断
        concat = torch.cat([img_feats, text_feats], dim=-1)
        raw_gate = self.gate_net(concat)  # (B, N, C)

        # 3. 【核心适配】融合外部去噪信号 (Explicit Denoising)
        if reliability is not None:
            # 逻辑：最终门控 = 内部意愿 * 外部置信度
            # 如果 DPG 说这个通道是噪音 (reliability -> 0)，
            # 那么无论内部 gate 想要多少，最后都得变成 0 (不注入文本)。
            final_gate = raw_gate * reliability
        else:
            final_gate = raw_gate

        # 4. 门控融合
        # Out = (1 - g) * V + g * T
        # 如果 final_gate 为 0 (噪音)，则 Out = V (完全保留原图，不受干扰)
        fused = (1 - final_gate) * v_prime + final_gate * t_prime

        # 5. 残差连接
        out = self.ln(img_feats + fused)

        return out