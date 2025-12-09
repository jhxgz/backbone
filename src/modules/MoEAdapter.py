import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEAdapter(nn.Module):
    """
    轻量级 MoE Adapter (混合专家适配器)

    设计思路：
    - 替代普通的 AdapterResidual (AR)。
    - 通过动态路由 (Dynamic Routing) 让不同的视觉 Token 激活不同的专家网络。
    - 增加模型容量（Capacity）但不显著增加推理计算量（FLOPs）。
    """

    def __init__(self, dim, num_experts=4, top_k=2, down_ratio=4, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 1. 路由器 (Router): 决定每个 Token 去哪个专家
        self.router = nn.Linear(dim, num_experts)

        # 2. 专家组 (Experts): 并行的轻量级 MLP
        # 这里的 down_ratio 可以设大一点 (如 4 或 8)，保持参数量可控
        self.experts = nn.ModuleList([
            MLPBlock(dim, down_ratio, dropout)
            for _ in range(num_experts)
        ])

        # 3. 归一化
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: [B, N, D]
        """
        residual = x
        x_norm = self.ln(x)  # [B, N, D]

        # A. 计算路由概率
        # logits: [B, N, num_experts]
        router_logits = self.router(x_norm)

        # B. 选取 Top-K 专家
        # indices: [B, N, k], weights: [B, N, k]
        routing_weights, selected_indices = torch.topk(router_logits, self.top_k, dim=-1)

        # C. 归一化权重 (Softmax)
        routing_weights = F.softmax(routing_weights, dim=-1)

        # D. 执行专家计算 (Soft Routing)
        # 由于 PyTorch 的动态特性，且 num_experts 较小 (4-8)，我们遍历专家比 gather/scatter 更快且易读

        final_output = torch.zeros_like(x_norm)

        for expert_idx in range(self.num_experts):
            # 1. 找出哪些 token 选中了当前专家 (mask)
            # selected_indices 中是否包含 expert_idx?
            # mask: [B, N] (Bool) -> 扩展为 [B, N, 1]
            is_selected = (selected_indices == expert_idx).any(dim=-1, keepdim=True)

            # 如果 batch 里没人选这个专家，跳过计算 (节省算力)
            if not is_selected.any():
                continue

            # 2. 获取该专家对这些 token 的权重
            # 我们需要从 routing_weights 中找到对应 expert_idx 的权重值
            # 构造一个 mask 来提取权重: [B, N, k]
            weight_mask = (selected_indices == expert_idx)  # [B, N, k]
            # 提取权重并求和 (因为每个 token 对同一个专家最多选一次) -> [B, N, 1]
            expert_weight = (routing_weights * weight_mask.float()).sum(dim=-1, keepdim=True)

            # 3. 专家前向传播
            # 这里我们对所有 token 都跑一遍专家 (为了并行效率)，然后用 mask 过滤
            # 只有被选中的 token 才会获得非零更新
            expert_out = self.experts[expert_idx](x_norm)

            # 4. 累加结果
            final_output += is_selected.float() * expert_weight * expert_out

        # E. 残差连接
        return residual + final_output


class MLPBlock(nn.Module):
    """基础的瓶颈层 (Bottleneck MLP)"""

    def __init__(self, dim, down_ratio=4, dropout=0.1):
        super().__init__()
        hidden_dim = dim // down_ratio
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)