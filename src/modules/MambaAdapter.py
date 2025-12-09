import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaAdapter(nn.Module):
    """
    轻量级 Mamba 视觉适配器 (Pure PyTorch Implementation)

    优势：
    1. 零依赖：不需要安装 mamba-ssm 和 causal-conv1d。
    2. 易部署：在任何安装了 PyTorch 的机器上都能跑（包括 Windows/Mac）。
    3. 针对短序列优化：对于 CLIP 的 49/196 个 Token，纯 PyTorch 实现效率极高。
    """

    def __init__(
            self,
            dim: int,
            d_state: int = 16,  # SSM 状态维度
            d_conv: int = 4,  # 局部卷积宽度
            expand: int = 2,  # 维度扩展倍数
            dropout: float = 0.1,
            num_layers: int = 1
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                MambaBlock(
                    d_model=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout
                )
            )

        # 最后的输出门控
        self.gate = nn.Parameter(torch.ones(dim) * 1e-3)

    def forward(self, x):
        """
        Input:  (B, N, D)
        Output: (B, N, D)
        """
        residual = x

        for layer in self.layers:
            # Mamba Block: Norm -> Mamba -> Residual
            out = layer(x)
            x = x + out

        # 门控融合
        g = torch.sigmoid(self.gate)
        output = residual + g * (x - residual)
        return output


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv
        self.d_state = d_state

        # 1. 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 2. 1D 卷积 (深度卷积)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # 3. 选择性投影 (x -> B, C, delta)
        # x_proj takes in `x` and outputs [dt, B, C]
        self.x_proj = nn.Linear(self.d_inner, (self.d_state * 2) + self.d_model, bias=False)

        # 4. dt 投影 (delta)
        self.dt_proj = nn.Linear(self.d_model, self.d_inner, bias=True)

        # 5. A (S4 参数) - Log 参数化
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 6. 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, u):
        """
        u: (B, L, D)
        """
        B, L, D = u.shape
        residual = u
        u = self.norm(u)

        # 1. 投影到内部维度 (B, L, 2*D_inner)
        xz = self.in_proj(u)
        x, z = xz.chunk(2, dim=-1)  # (B, L, D_inner)

        # 2. 卷积 (需要转置: B, D, L)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]  # Causal padding
        x = x.transpose(1, 2)

        x = self.act(x)

        # 3. SSM 核心 (Scan)
        y = self.ssm(x)

        # 4. 门控与输出
        y = y * self.act(z)
        out = self.out_proj(y)
        out = self.dropout(out)
        return out

    def ssm(self, x):
        """
        手动实现选择性扫描 (Selective Scan)
        x: (B, L, D_inner)
        """
        B, L, D_inner = x.shape

        # 计算 Delta, B, C
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)

        dt, B_param, C_param = torch.split(x_dbl, [self.d_model, self.d_state, self.d_state], dim=-1)
        # dt: (B, L, d_model) -> (B, L, D_inner)
        dt = self.dt_proj(dt)
        dt = F.softplus(dt)  # 保证 delta > 0

        # 参数离散化
        # A: (D_inner, d_state)
        A = -torch.exp(self.A_log)

        # 离散化 A -> dA: (B, L, D_inner, d_state)
        # dA = exp(delta * A)
        dA = torch.exp(torch.einsum('bld,dn->bldn', dt, A))

        # 离散化 B -> dB: (B, L, D_inner, d_state)
        # dB = delta * B
        dB = torch.einsum('bld,bln->bldn', dt, B_param)

        # 扫描 (Scan) - 简单的串行循环
        # 对于 L=49 或 196，这个循环非常快，不需要 CUDA
        h = torch.zeros(B, D_inner, self.d_state, device=x.device)
        ys = []

        for i in range(L):
            # h_t = dA * h_{t-1} + dB * x_t
            # x[i]: (B, D_inner) -> (B, D_inner, 1)
            h = dA[:, i] * h + dB[:, i] * x[:, i].unsqueeze(-1)

            # y_t = C * h_t
            # C[i]: (B, d_state) -> (B, 1, d_state)
            y = torch.einsum('bdn,bn->bd', h, C_param[:, i])
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (B, L, D_inner)

        # 加上 D * x (Skip connection)
        y = y + x * self.D
        return y