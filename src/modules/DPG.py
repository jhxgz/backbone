import torch
import torch.nn as nn
import torch.nn.functional as F


class DetailOrientedPromptGenerator(nn.Module):
    """
    DPG (Detail-Oriented Prompt Generation) 模块
    基于 DoPL 论文 (ACL 2025) 复现。

    核心思想：
    利用“信息熵”来自动识别文本和视觉特征中的“共同兴趣点”。
    熵越低，说明对齐越明确；熵越高，说明是噪音。
    通过给低熵区域分配高权重，实现无参数化的去噪和增强。
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.epsilon = 1e-8  # 防止 log(0)

        # 为了适配特征空间，我们只加一层极轻量的线性投影
        # 论文中是直接作用于 Prompt，但为了特征融合的稳定性，加一个 Linear 是工程上的最佳实践
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # 残差连接后的 LayerNorm
        self.ln = nn.LayerNorm(dim)

    def forward(self, text_feats, visual_feats):
        """
        Args:
            text_feats: (B, Lt, D) - 文本特征 (Query)
            visual_feats: (B, Lv, D) - 视觉特征 (Key/Value)
        Returns:
            enhanced_text: (B, Lt, D)
        """
        # 1. 计算相关性矩阵 R (B, Lt, Lv) [对应论文公式 2]
        # 使用点积作为相似度度量
        sim_matrix = torch.matmul(text_feats, visual_feats.transpose(-1, -2))
        # 缩放点积 (Scale Dot-Product) 防止数值过大
        sim_matrix = sim_matrix / (self.dim ** 0.5)

        # 2. 归一化得到概率分布 N (text -> visual) [对应论文公式 3]
        # 表示：对于每个文本 token，它对应各个视觉 token 的概率分布
        prob_t2v = F.softmax(sim_matrix, dim=-1)  # (B, Lt, Lv)

        # 3. 计算熵矩阵 H (Entropy) [对应论文公式 5]
        # H = - p * log(p)
        # 熵越低，说明该文本 token 极其确信地指向了某个视觉区域（有效信息）
        # 熵越高，说明该文本 token 和谁都沾点边（可能是噪音或常用词）
        entropy_matrix = - prob_t2v * torch.log(prob_t2v + self.epsilon)

        # 4. 计算 summed entropy [对应论文公式 7]
        # 对视觉维度求和，得到每个文本 token 的总不确定性
        entropy_scores = entropy_matrix.sum(dim=-1)  # (B, Lt)

        # 5. 生成对齐权重 W [对应论文公式 9]
        # 负熵越大（即熵越小），权重越大。
        # 这里我们在 Batch 内部或者 Token 序列内部进行 Softmax 归一化
        # 为了突出序列中重要的词，我们在 dim=1 (Lt) 上做 Softmax
        alignment_weights = F.softmax(-entropy_scores, dim=-1)  # (B, Lt)

        # 6. 利用权重增强特征 [对应论文公式 11]
        # 将权重作用于文本特征本身，或者通过视觉特征加权求和
        # DoPL 论文是增强 Prompt，这里我们用来从 Visual 中提取信息

        # 变体实现：使用对齐权重从 Visual 中提取“最确信”的信息
        # context = weights * (Prob * Visual)
        # 但为了保留序列长度，我们采用论文思路：增强 text_feats

        # 将权重扩展维度以便广播: (B, Lt, 1)
        w_expanded = alignment_weights.unsqueeze(-1)

        # 这里的实现有一个灵活点：
        # 论文是用权重去增强 Prompt 参数。我们这里是用权重去“门控”视觉信息流入。
        # 我们计算一个 Context：用 prob_t2v 加权 Visual，得到每个文本词对应的视觉中心
        visual_context = torch.matmul(prob_t2v, visual_feats)  # (B, Lt, D)

        # 然后用 熵权重 对这个 Context 进行过滤：只有低熵（准）的 Context 才能保留
        refined_context = visual_context * w_expanded  # (B, Lt, D)

        # 7. 投影与残差融合 [对应论文公式 13]
        out = self.proj(refined_context)
        out = self.ln(text_feats + out)

        return out