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
                nn.init.constant_(last_linear.bias, 0.0)

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
        # === 准备 Reliability (只做维度对齐，暂不使用) ===
        global_reliability = None
        if reliability is not None:
            # 统一维度处理逻辑 (保持你原有的处理，确保形状为 (B, 1, 1))
            if reliability.dim() == 3:
                if reliability.shape[-1] == 1:
                    global_reliability = reliability.mean(dim=1, keepdim=True)
                else:
                    global_reliability = reliability.mean(dim=(1, 2), keepdim=True)
            elif reliability.dim() == 2:
                global_reliability = reliability.mean(dim=1, keepdim=True).unsqueeze(-1)
            elif reliability.dim() == 1:
                global_reliability = reliability.mean(keepdim=True).unsqueeze(-1).unsqueeze(-1)
            else:
                global_reliability = reliability

        # 3. 生成软门控 (Soft Gate)
        # [修改] 让 gate 正常生成，不受 reliability 干扰输入分布
        # gate_input 应该反映纯粹的 V-T 相关性
        gate_input = img_feats * torch.sigmoid(relevance_score)
        gate = self.gate_gen(gate_input)  # (B, Ni, 1)

        # 4. [新增] 后置可靠性控制 (Post-Gate Modulation)
        # 正确逻辑：如果 Text 可靠，使用计算出的 Gate；
        #          如果 Text 不可靠 (reliability -> 0)，Gate -> 0。
        #          因为最后是 Residual 连接，Gate -> 0 意味着 "Output = Residual" (保留原图)
        #          这避免了错误地剔除图像特征。
        if global_reliability is not None:
            # 确保维度匹配，进行广播乘法
            gate = gate * global_reliability

        # 5. 过滤视觉特征
            refined_img = img_feats * gate

        # 6. 残差连接 + Norm
        # 如果 reliability 很低，gate 接近 0，refined_img 接近 0
        # 结果 out 接近 ln(residual)，即无损保留原始图像特征
        out = self.ln(residual + self.dropout(refined_img))

        return out