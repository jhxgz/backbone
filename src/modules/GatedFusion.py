import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        # 投影层：将拼接后的 2*dim 映射回 dim
        self.proj_content = nn.Linear(dim * 2, dim)
        # 门控层：同样将 2*dim 映射回 dim
        self.proj_gate = nn.Linear(dim * 2, dim)

        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_feats, text_feats, reliability=None):
        # 1. 拼接 (继承 Baseline 的优点)
        concat_input = torch.cat([img_feats, text_feats], dim=-1)  # [B, N, 2D]

        # 2. 计算内容流 (Content Stream)
        content = self.proj_content(concat_input)

        # 3. 计算门控流 (Gate Stream)
        gate = torch.sigmoid(self.proj_gate(concat_input))

        # 4. 注入 Reliability
        if reliability is not None:
            gate = gate * reliability  # 这里可以直接乘，因为基座是 Concat
            if self.training and torch.rand(1).item() < 0.01:
                print(f"[Fusion Monitor] Final Effective Gate Mean: {gate.mean().item():.4f}")

        # 5. 门控乘法 (Element-wise multiplication)
        out = content * gate

        # 6. 残差连接 (可选，建议加上)
        out = self.ln(img_feats + self.dropout(out))

        return out