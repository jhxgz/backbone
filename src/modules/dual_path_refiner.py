# coding=utf-8
"""
双路径增强模块 (Dual Path Refiner)
在 CLIP image encoder 输出后，通过双路径残差形式增强特征表示。

设计动机：
- LayerNorm 提供稳定的特征归一化
- AdapterResidual 路径：轻量级残差适配器，增强局部特征表示
- AttentionPoolingRefiner 路径：注意力池化精炼器，增强全局特征表示
- 双路径并行设计可以同时利用局部和全局信息
- 可学习的 alpha1 和 alpha2 参数允许模型自动平衡两条路径的贡献
"""

import torch
import torch.nn as nn
from .AR import AdapterResidual
from .APR import AttentionPoolingRefiner


class DualPathRefiner(nn.Module):
    """
    双路径增强模块
    
    输入: x (B, N, C)
    输出: x_out (B, N, C)
    
    流程:
    1. v = LayerNorm(x)
    2. y1 = AdapterResidual(v)
    3. y2 = AttentionPoolingRefiner(v)
    4. x_out = x + alpha1 * y1 + alpha2 * y2
    """
    
    def __init__(
        self,
        dim: int,
        # AdapterResidual 参数
        adapter_down_ratio: int = 4,
        adapter_dropout: float = 0.1,
        adapter_use_gate: bool = True,
        # AttentionPoolingRefiner 参数
        apr_n_queries: int = 1,
        apr_n_heads: int = 8,
        apr_proj_back: bool = True,
        apr_dropout: float = 0.1,
        # 双路径权重参数
        alpha1_init: float = 0.1,
        alpha2_init: float = 0.1,
    ):
        """
        Args:
            dim: 特征维度 (C)
            adapter_down_ratio: AdapterResidual 的降维比例
            adapter_dropout: AdapterResidual 的 dropout 率
            adapter_use_gate: AdapterResidual 是否使用门控机制
            apr_n_queries: AttentionPoolingRefiner 的查询数量
            apr_n_heads: AttentionPoolingRefiner 的注意力头数
            apr_proj_back: AttentionPoolingRefiner 是否投影回 token 空间
            apr_dropout: AttentionPoolingRefiner 的 dropout 率
            alpha1_init: alpha1 的初始值
            alpha2_init: alpha2 的初始值
        """
        super().__init__()
        self.dim = dim
        
        # LayerNorm：对输入特征进行归一化
        self.ln = nn.LayerNorm(dim, eps=1e-6)
        
        # 路径1：AdapterResidual - 轻量级残差适配器
        self.adapter_residual = AdapterResidual(
            dim=dim,
            down_ratio=adapter_down_ratio,
            dropout=adapter_dropout,
            use_gate=adapter_use_gate,
        )
        
        # 路径2：AttentionPoolingRefiner - 注意力池化精炼器
        self.attention_pooling_refiner = AttentionPoolingRefiner(
            dim=dim,
            n_queries=apr_n_queries,
            n_heads=apr_n_heads,
            proj_back=apr_proj_back,
            dropout=apr_dropout,
        )
        
        # 可学习的权重参数：控制两条路径对最终输出的贡献
        # 初始化为小值，确保训练初期以原始特征为主
        self.alpha1 = nn.Parameter(torch.tensor(alpha1_init, dtype=torch.float32))
        self.alpha2 = nn.Parameter(torch.tensor(alpha2_init, dtype=torch.float32))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 (B, N, C)
            
        Returns:
            x_out: 增强后的特征 (B, N, C)
        """
        # 步骤1: LayerNorm 归一化
        v = self.ln(x)  # (B, N, C)
        
        # 步骤2: 并行执行两条路径
        y1 = self.adapter_residual(v)  # (B, N, C) - 路径1：残差适配器
        y2 = self.attention_pooling_refiner(v)  # (B, N, C) - 路径2：注意力池化精炼器
        
        # 步骤3: 双路径残差合并
        # 保留原始特征 x，加上两条路径的加权贡献
        x_out = x + self.alpha1 * y1 + self.alpha2 * y2  # (B, N, C)
        
        return x_out

