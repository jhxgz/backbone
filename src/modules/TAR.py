import torch
import torch.nn as nn

class VisualGuidedTAR(nn.Module):
    """
    Visual-Guided Text Adapter Residual (VG-TAR) 模块。
    使用跨模态交互，通过图像特征精炼文本特征。
    
    输入：
        - text_feats: (B, T_text, C) - 文本特征作为 Query
        - visual_feats: (B, T_vis, C) - 视觉特征作为 Key/Value
    
    输出：
        - (B, T_text, C) - 增强后的文本特征
    
    使用方式：
        vg_tar = VisualGuidedTAR(dim=C, num_heads=8, dropout=0.1, use_gate=True, init_gate=-4.0)
        out = vg_tar(text_feats, visual_feats)
    
    关键设计要点：
        1. 门控初始化：init_gate 默认值为 -4.0，确保训练初期 Sigmoid(gate) 接近 0，
           模块主要依赖原始文本特征（残差连接），避免过度过滤。
        2. 特征投影：Cross-Attention 的 Q, K, V 投影层是可训练的且独立初始化，
           不会与 CLIP 编码器权重共享，能够适配 AR+APR 增强后的视觉特征。
        3. LayerNorm 结构（稳定训练）：
           - Pre-LN 架构：在 Cross-Attention 和 FFN 之前使用 LayerNorm，更稳定的梯度流
           - Post-LN 架构：在残差连接之后使用 LayerNorm，稳定特征分布
           - Visual 输入归一化：对 visual_feats 进行 LayerNorm，防止跨模态交互中的梯度爆炸
        4. 梯度稳定性：配合全局梯度裁剪（max_grad_norm=1.0），确保训练稳定
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, use_gate=True, init_gate=-4.0, ffn_dim=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_gate = use_gate
        
        # Pre-LayerNorm for Cross-Attention (更稳定的梯度流)
        self.ln_pre_attn = nn.LayerNorm(dim)
        
        # Multihead Cross-Attention (MHCA)
        # Query来自text_feats, Key/Value来自visual_feats
        # 注意：nn.MultiheadAttention 会自动创建可训练的 Q, K, V 投影层
        # 这些投影层是独立的，不会与 CLIP 编码器权重共享
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # LayerNorm after cross-attention (Post-LN for residual)
        self.ln1 = nn.LayerNorm(dim)
        
        # Dropout after cross-attention output (增强正则化，防止过拟合)
        self.dropout_attn = nn.Dropout(dropout)
        
        # Pre-LayerNorm for FFN
        self.ln_pre_ffn = nn.LayerNorm(dim)
        
        # Feed Forward Network (FFN)
        if ffn_dim is None:
            ffn_dim = dim * 4  # 标准Transformer的FFN维度
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Dropout after FFN output (增强正则化，防止过拟合)
        self.dropout_ffn = nn.Dropout(dropout)
        
        # LayerNorm after FFN (Post-LN for residual)
        self.ln2 = nn.LayerNorm(dim)
        
        # LayerNorm for visual_feats input (稳定跨模态交互)
        self.ln_visual = nn.LayerNorm(dim)
        
        # Gate mechanism (应用在cross-attention输出上)
        # 初始化门控参数为较大的负数，确保训练初期 Sigmoid(gate) 接近 0
        # 这样模块在训练初期主要依赖原始文本特征（残差连接）
        if use_gate:
            self.gate = nn.Parameter(torch.full((dim,), init_gate))
        else:
            self.gate = None
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 初始化 Cross-Attention 的 Q, K, V 投影层
        # nn.MultiheadAttention 内部结构：
        # - in_proj_weight: [3 * embed_dim, embed_dim] (Q, K, V 拼接)
        # - in_proj_bias: [3 * embed_dim]
        # - out_proj: Linear(embed_dim, embed_dim)
        
        # 初始化 Q, K, V 投影层权重（使用 Xavier 初始化）
        if hasattr(self.cross_attn, 'in_proj_weight') and self.cross_attn.in_proj_weight is not None:
            # in_proj_weight 形状: [3 * embed_dim, embed_dim]
            # 分别初始化 Q, K, V 三个部分
            embed_dim = self.dim
            nn.init.xavier_uniform_(self.cross_attn.in_proj_weight[:embed_dim, :])  # Q 投影
            nn.init.xavier_uniform_(self.cross_attn.in_proj_weight[embed_dim:2*embed_dim, :])  # K 投影
            nn.init.xavier_uniform_(self.cross_attn.in_proj_weight[2*embed_dim:, :])  # V 投影
            
        # 初始化 Q, K, V 投影层偏置
        if hasattr(self.cross_attn, 'in_proj_bias') and self.cross_attn.in_proj_bias is not None:
            nn.init.constant_(self.cross_attn.in_proj_bias, 0.)
            
        # 初始化输出投影层
        if hasattr(self.cross_attn, 'out_proj'):
            nn.init.xavier_uniform_(self.cross_attn.out_proj.weight)
            if self.cross_attn.out_proj.bias is not None:
                nn.init.constant_(self.cross_attn.out_proj.bias, 0.)
        
        # 初始化FFN的线性层
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.)
    
    def forward(self, text_feats, visual_feats):
        """
        Args:
            text_feats: (B, T_text, C) - 文本特征作为Query
            visual_feats: (B, T_vis, C) - 视觉特征作为Key/Value
        
        Returns:
            out: (B, T_text, C) - 增强后的文本特征
        """
        # Step 0: 对 visual_feats 进行 LayerNorm（稳定跨模态交互，防止梯度爆炸）
        visual_feats_norm = self.ln_visual(visual_feats)
        
        # 保存原始text_feats用于残差连接
        residual = text_feats
        
        # Step 1: Pre-LayerNorm + Multihead Cross-Attention
        # 使用 Pre-LN 架构，更稳定的梯度流
        text_feats_norm = self.ln_pre_attn(text_feats)
        cross_attn_output, _ = self.cross_attn(
            query=text_feats_norm,
            key=visual_feats_norm,
            value=visual_feats_norm
        )  # (B, T_text, C)
        
        # Step 2: 应用门控机制（如果启用）
        if self.use_gate:
            g = torch.sigmoid(self.gate)  # (C,)
            cross_attn_output = cross_attn_output * g.view(1, 1, -1)
        
        # Step 2.5: 应用 Dropout 到 Cross-Attention 输出（增强正则化）
        cross_attn_output = self.dropout_attn(cross_attn_output)
        
        # Step 3: 第一次残差连接 + Post-LayerNorm
        out = self.ln1(residual + cross_attn_output)  # (B, T_text, C)
        
        # Step 4: Pre-LayerNorm + Feed Forward Network
        out_norm = self.ln_pre_ffn(out)
        ffn_output = self.ffn(out_norm)  # (B, T_text, C)
        
        # Step 4.5: 应用 Dropout 到 FFN 输出（增强正则化）
        ffn_output = self.dropout_ffn(ffn_output)
        
        # Step 5: 第二次残差连接 + Post-LayerNorm
        out = self.ln2(out + ffn_output)  # (B, T_text, C)
        
        return out


# 保留原TextAdapterResidual类以保持向后兼容（如果需要）
class TextAdapterResidual(nn.Module):
    """
    Text-side Adapter Residual (已弃用，保留用于向后兼容)。
    请使用 VisualGuidedTAR 替代。
    """
    def __init__(self, dim, down_ratio=8, dropout=0.1, use_gate=True, init_gate=-2.0):
        super().__init__()
        mid = max(8, dim // down_ratio)
        self.ln = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, mid)
        self.act = nn.GELU()
        self.up = nn.Linear(mid, dim)
        self.dropout = nn.Dropout(dropout)
        self.use_gate = use_gate
        if use_gate:
            # per-channel gate
            self.gate = nn.Parameter(torch.ones(dim) * init_gate)
        else:
            self.gate = None

        # 初始化
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.constant_(self.down.bias, 0.)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.constant_(self.up.bias, 0.)

    def forward(self, x):
        """
        x: (B, T, C) or (B, C)
        returns same shape
        """
        orig_shape = x.shape
        if x.dim() == 2:
            # (B, C) -> make (B, 1, C) to reuse code
            x_in = x.unsqueeze(1)
            squeeze_out = True
        else:
            x_in = x
            squeeze_out = False

        # LN + bottleneck MLP applied token-wise
        y = self.ln(x_in)
        y = self.down(y)
        y = self.act(y)
        y = self.up(y)
        y = self.dropout(y)

        if self.use_gate:
            g = torch.sigmoid(self.gate)  # (C,)
            y = y * g.view(1, 1, -1)

        out = x_in + y  # residual
        if squeeze_out:
            return out.squeeze(1)
        return out
