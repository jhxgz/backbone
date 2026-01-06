import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelWiseDPG(nn.Module):
    """
    [修改版] Channel-wise DPG (去噪提示门控)

    改进功能：
    1. 结合 DoPL 的熵理论 (Scalar Gate) 和 新增的 Channel MLP (Vector Gate)。
    2. 输出 explicit denoising signal (alpha) 供 GateFusion 使用。
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.epsilon = 1e-8  # 防止 log(0)

        # 1. 特征投影层 (保持不变)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.ln = nn.LayerNorm(dim)

        # 2. 【新增】通道级门控网络 (Channel-wise Gating MLP)
        # 输入: Text(D) + Visual_Context(D) = 2D
        # 输出: Gate(D) -> 每个通道独立的 0~1 开关
        self.channel_gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),  # 降维减少参数
            nn.ReLU(),
            nn.Linear(dim // 4, dim),  # 升维回 D
            nn.Sigmoid()  # 关键：输出 0~1 的门控值
        )
        
        # 初始化最后一层偏置，确保训练初期 Sigmoid 输出接近 0.88，让门控默认处于"开启"状态
        nn.init.constant_(self.channel_gate_mlp[-2].bias, 2.0)

    def forward(self, text_feats, visual_feats):
        """
        Args:
            text_feats: (B, Lt, D) - 检索文本特征
            visual_feats: (B, Lv, D) - 图像特征
        Returns:
            refined_text: (B, Lt, D) - 增强后的文本特征
            alpha: (B, Lt, D) - 【新增】去噪门控信号
        """
        # --- Part 1: DoPL 基础流程 (计算基于熵的标量权重) ---

        # 1. 计算相关性 & 概率分布
        sim_matrix = torch.matmul(text_feats, visual_feats.transpose(-1, -2))
        sim_matrix = sim_matrix / (self.dim ** 0.5)
        prob_t2v = F.softmax(sim_matrix, dim=-1)  # (B, Lt, Lv)

        # 2. 计算熵 (Entropy)
        entropy_matrix = - prob_t2v * torch.log(prob_t2v + self.epsilon)
        entropy_scores = entropy_matrix.sum(dim=-1)  # (B, Lt)

        # 3. 生成标量权重 (Scalar Weight) - DoPL 的核心
        # 熵越低 -> 权重越大 (置信度高)
        # (B, Lt) -> (B, Lt, 1)
        scalar_weight = F.softmax(-entropy_scores, dim=-1).unsqueeze(-1)

        # 4. 获取视觉上下文
        visual_context = torch.matmul(prob_t2v, visual_feats)  # (B, Lt, D)

        # --- Part 2: 【改进】Channel-wise Gating & Alpha 计算 ---

        # 1. 计算通道级门控向量
        # 我们依据“文本”和“它找到的视觉内容”是否匹配来决定
        gate_input = torch.cat([text_feats, visual_context], dim=-1)  # (B, Lt, 2D)
        vector_gate = self.channel_gate_mlp(gate_input)  # (B, Lt, D)

        # 2. 融合双重门控，得到最终去噪信号 alpha
        # alpha = 整体置信度 (Scalar) * 通道选择性 (Vector)
        alpha = scalar_weight * vector_gate  # (B, Lt, D)

        # --- Part 3: 特征输出 ---

        # 1. 利用 alpha 进行显式过滤 (Refinement)
        # 这一步保证了流向 Residual 连接的信息是干净的
        refined_context = visual_context * alpha

        # 2. 残差连接
        out = self.proj(refined_context)
        refined_text = self.ln(text_feats + out)

        # 【关键】必须把 alpha 返回出去，给 GateFusion 用！
        return refined_text, alpha