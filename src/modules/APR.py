import torch
import torch.nn as nn
class AttentionPoolingRefiner(nn.Module):
    def __init__(self, dim, n_queries=1, n_heads=8, proj_back=True, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.n_queries = n_queries
        self.dim = dim
        self.proj_back = proj_back
        self.query = nn.Parameter(torch.randn(n_queries, dim))  # learnable queries
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        if proj_back:
            # project global queries back to token space via cross-attn
            self.kv_to_token_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
            self.out_ln = nn.LayerNorm(dim)
            self.out_fc = nn.Linear(dim, dim)
        else:
            self.out_fc = nn.Linear(n_queries * dim, dim)
            self.out_ln = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        y = self.ln(x)
        queries = self.query.unsqueeze(0).expand(B, -1, -1)  # (B, n_q, D)
        # queries attend to tokens (K=V=x)
        pooled, _ = self.multihead_attn(queries, y, y)  # (B, n_q, D)
        if self.proj_back:
            # use pooled as queries to attend to tokens and produce token-wise updates
            # pooled -> tokens
            token_update, _ = self.kv_to_token_attn(y, pooled, pooled)  # keys=pooled? we want pooled as K/V, queries are tokens
            # Above call uses query=y, key=pooled, value=pooled: (B,N,D)
            out = self.out_ln(token_update)
            out = self.out_fc(out)
            return out
        else:
            # broadcast pooled summary to per-token update
            pooled_flat = pooled.reshape(B, -1)  # (B, n_q*D)
            out = self.out_fc(pooled_flat).unsqueeze(1).expand(B, N, D)
            out = self.out_ln(out)
            return out
