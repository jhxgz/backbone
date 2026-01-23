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

        # 自定义初始化：将最后一个线性层（Sigmoid 前一层，索引为 -2）的 bias 初始化为 -3.0
        # 目的：使 Sigmoid 的初始输出接近 0，让模块在训练初期近似于恒等映射
        if len(self.gate_gen) >= 2:
            last_linear = self.gate_gen[-2]  # 获取最后一个线性层（Sigmoid 前一层）
            if isinstance(last_linear, nn.Linear) and last_linear.bias is not None:
                nn.init.constant_(last_linear.bias, -3.0)

        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_feats, text_feats, reliability=None):
        """
        img_feats:  (B, N_img, D)
        text_feats: (B, N_text, D)  <-- 来自 DPG 的输出
        reliability: (B, N_text, 1) or (B, N_text) or (B, 1, 1) or None - DPG 计算出的 alpha，表示文本的可靠性
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

        # 2.5. 可靠性加权：如果文本不可靠，降低相关性评分
        if reliability is not None:
            # 计算文本的全局置信度（对 reliability 进行 mean pooling）
            if reliability.dim() == 3:  # (B, N_text, 1) or (B, N_text, C)
                # 如果是 3D，先对文本维度求平均
                if reliability.shape[-1] == 1:
                    # (B, N_text, 1) -> (B, 1, 1)
                    global_reliability = reliability.mean(dim=1, keepdim=True)
                else:
                    # (B, N_text, C) -> (B, 1, 1)
                    global_reliability = reliability.mean(dim=(1, 2), keepdim=True)
            elif reliability.dim() == 2:  # (B, N_text)
                # (B, N_text) -> (B, 1, 1)
                global_reliability = reliability.mean(dim=1, keepdim=True).unsqueeze(-1)
            elif reliability.dim() == 1:  # (B,)
                # (B,) -> (B, 1, 1)
                global_reliability = reliability.mean(keepdim=True).unsqueeze(-1).unsqueeze(-1)
            else:
                # 已经是 (B, 1, 1) 或类似形状
                global_reliability = reliability
            
            # 确保维度匹配 relevance_score: (B, Ni, 1)
            if global_reliability.shape != relevance_score.shape:
                # 扩展 global_reliability 到 (B, Ni, 1)
                global_reliability = global_reliability.expand_as(relevance_score)
            
            # 使用可靠性对 relevance_score 进行加权（乘法）
            relevance_score = relevance_score * global_reliability

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