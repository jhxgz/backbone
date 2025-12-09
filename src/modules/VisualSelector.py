import torch
import torch.nn as nn
import torch.nn.functional as F


class TextGuidedVisualSelector(nn.Module):
    """
    轻量级视觉选择器 (ADEM-VL 风格)
    利用 DPG 去噪后的文本特征，反向过滤视觉特征中的无关背景。

    互补性：
    - DPG: 过滤文本噪音 (Text Denoising)
    - Selector: 过滤视觉噪音 (Visual Denoising)
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim

        # 简单的注意力打分器
        # 将 text 和 visual 映射到同一空间计算相关性
        self.scale = dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)  # Visual -> Query
        self.k_proj = nn.Linear(dim, dim, bias=False)  # Text -> Key

        # 门控生成器
        self.gate_gen = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_feats, text_feats):
        """
        img_feats:  (B, N_img, D)
        text_feats: (B, N_text, D)  <-- 来自 DPG 的输出
        """
        residual = img_feats

        # 1. 计算 Visual-Text 相关性矩阵
        # Q = Visual, K = Text
        # 我们想知道每个 visual token 对 text 的重要程度
        Q = self.q_proj(img_feats)  # (B, Ni, D)
        K = self.k_proj(text_feats)  # (B, Nt, D)

        # Attention map: (B, Ni, Nt)
        attn = torch.matmul(Q, K.transpose(-1, -2)) * self.scale

        # 2. 聚合文本信息到视觉侧
        # 对于每个 visual token，找到它最相关的 text 及其强度
        # Max-Pooling over text dimension: 只关心这个 visual token 是否与 *任意* text token 相关
        relevance_score, _ = attn.max(dim=-1, keepdim=True)  # (B, Ni, 1)

        # 3. 生成软门控 (Soft Gate)
        # 这里的 relevance_score 指示了视觉区域的重要性
        # 我们把它变换成 0~1 的 gate
        # 为了更平滑，可以加一层 MLP (gate_gen)
        # 或者直接用 sigmoid(relevance_score)

        # 这里我们用 img_feats 结合 relevance_score 来生成 gate
        # 意思是：不仅看相关性，也看视觉内容本身 (Context-aware)
        gate_input = img_feats * torch.sigmoid(relevance_score)
        gate = self.gate_gen(gate_input)  # (B, Ni, 1)

        # 4. 过滤视觉特征
        refined_img = img_feats * gate

        # 5. 残差连接 + Norm
        out = self.ln(residual + self.dropout(refined_img))

        return out