import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelSE(nn.Module):
    """Channel squeeze-excite but supports sequence or spatial inputs."""
    def __init__(self, dim, reduction=4):
        super().__init__()
        mid = max(8, dim // reduction)
        self.fc = nn.Sequential(
            nn.Linear(dim, mid),
            nn.GELU(),
            nn.Linear(mid, dim),
            nn.Sigmoid()
        )

    def forward(self, img_feats, prompt_feats=None, prompt_mask=None):
        # img_feats: [B, P, D]; prompt_feats: [B, L, D] or None
        B, P, D = img_feats.shape

        # If no prompt provided, simply return original features (optionally pass through small FF)
        if prompt_feats is None:
            # keep behavior consistent: apply channel attention + small residual ffb to keep param usage
            img_ca = self.ca_img(img_feats)
            combo = torch.cat([img_feats, img_ca, img_feats * img_ca, img_feats * 0.0], dim=-1)  # last term dummy
            fused = self.ffb(combo) if combo.shape[-1] == self.ffb[0].in_features else img_feats
            # gating fallback (no prompt)
            pooled_fused = fused.mean(dim=1)
            pooled_img = img_feats.mean(dim=1)
            g = self.gate(torch.cat([pooled_fused, pooled_img], dim=-1))
            g = g.unsqueeze(1)
            return g * fused + (1 - g) * img_feats

        # ensure prompt_feats on same device
        prompt_feats = prompt_feats.to(img_feats.device)

        B2, L, D2 = prompt_feats.shape
        if not (B == B2 and D == D2):
            # Try to be friendly: if dims mismatch, try linear projection (warn)
            raise ValueError(
                f"[AdaptedFFM] batch/dim mismatch img_feats {img_feats.shape} vs prompt_feats {prompt_feats.shape}")

        # Normalize/convert prompt_mask to boolean mask where True means valid
        if prompt_mask is None:
            kv_mask_bool = None
        else:
            if prompt_mask.dtype == torch.bool:
                kv_mask_bool = prompt_mask
            else:
                kv_mask_bool = (prompt_mask != 0)
            # ensure on correct device
            kv_mask_bool = kv_mask_bool.to(img_feats.device)

        # 1) Channel attention (emphasize important channels)
        img_ca = self.ca_img(img_feats)  # [B, P, D]
        prompt_ca = self.ca_prompt(prompt_feats)  # [B, L, D]

        # 2) Cross-attention: image queries prompt (inject prompt into each patch)
        img_attn_out, attn_ip = self.img2prompt(img_ca, prompt_ca, kv_mask=kv_mask_bool)  # [B, P, D]

        # 3) Cross-attention: prompt queries image (get prompt-aware summarization)
        prompt2img_out, attn_pi = self.prompt2img(prompt_ca, img_ca)  # [B, L, D]

        # 4) correlation enhancement: compute affinity [B, P, L]
        A = torch.matmul(self.affine_i(img_ca), self.affine_p(prompt_ca).transpose(1, 2)) / math.sqrt(D)
        A = torch.softmax(A, dim=-1)  # [B, P, L]
        ce = torch.matmul(A, prompt2img_out)  # [B, P, D]

        # 5) Build combination tensor per patch: [img, img_attn_out, ce, img * img_attn_out]
        combo = torch.cat([img_feats, img_attn_out, ce, img_feats * img_attn_out], dim=-1)  # [B, P, 4D]

        # 6) Fusion and reduce back to D
        fused = self.ffb(combo)  # [B, P, D]

        # 7) gating + residual
        pooled_fused = fused.mean(dim=1)  # [B, D]
        pooled_img = img_feats.mean(dim=1)  # [B, D]
        g = self.gate(torch.cat([pooled_fused, pooled_img], dim=-1))  # [B, D]
        g = g.unsqueeze(1)  # [B,1,D]
        out = g * fused + (1 - g) * img_feats  # [B, P, D]

        return out


class SimpleCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qk_scale=None, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = qk_scale or (self.head_dim ** -0.5)

    def forward(self, q_in, kv_in, kv_mask=None):
        # q_in: [B, Q, D]  kv_in: [B, K, D]
        B, Q, D = q_in.shape
        K = kv_in.shape[1]
        q = self.q(q_in).reshape(B, Q, self.num_heads, self.head_dim).permute(0,2,1,3)  # [B, H, Q, Hd]
        k = self.k(kv_in).reshape(B, K, self.num_heads, self.head_dim).permute(0,2,3,1)  # [B, H, Hd, K]
        v = self.v(kv_in).reshape(B, K, self.num_heads, self.head_dim).permute(0,2,1,3)  # [B, H, K, Hd]

        attn = (q @ k) * self.scale  # [B, H, Q, K]
        if kv_mask is not None:
            # kv_mask: [B, K] -> expand
            attn = attn.masked_fill(~kv_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v)  # [B, H, Q, Hd]
        out = out.permute(0,2,1,3).reshape(B, Q, D)  # [B, Q, D]
        out = self.out(out)
        out = self.dropout(out)
        return out, attn  # return attention optionally

class AdaptedFFM(nn.Module):
    """
    Cross-modal adaptation of the original FFM:
    - img_feats: [B, P, D]  (patch sequence)
    - prompt_feats: [B, L, D] (token sequence)
    returns: fused_img_feats: [B, P, D]
    """
    def __init__(self, dim, hidden=512, num_heads=4):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.num_heads = num_heads

        # Channel attention for each modality
        self.ca_img = ChannelSE(dim, reduction=4)
        self.ca_prompt = ChannelSE(dim, reduction=4)

        # Cross-attention blocks (image queries prompt; prompt queries image summary)
        self.img2prompt = SimpleCrossAttention(dim, num_heads=num_heads)
        self.prompt2img = SimpleCrossAttention(dim, num_heads=num_heads)

        # Correlation enhancement projections
        self.affine_i = nn.Linear(dim, dim)
        self.affine_p = nn.Linear(dim, dim)

        # Fusion MLP (similar to original FFB but simpler / lighter)
        self.ffb = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.LayerNorm(dim)
        )

        # gate
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        # optional small conv projection if you need spatial ops (kept unused)
        # self.spat_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, img_feats, prompt_feats, prompt_mask=None):
        # img_feats: [B, P, D]; prompt_feats: [B, L, D]
        B, P, D = img_feats.shape
        B2, L, D2 = prompt_feats.shape
        assert B == B2 and D == D2, "Batch or dim mismatch"

        # 1) Channel attention (emphasize important channels)
        img_ca = self.ca_img(img_feats)       # [B, P, D]
        prompt_ca = self.ca_prompt(prompt_feats)  # [B, L, D]

        # 2) Cross-attention: image queries prompt (inject prompt into each patch)
        img_attn_out, attn_ip = self.img2prompt(img_ca, prompt_ca, kv_mask=prompt_mask)  # [B, P, D]

        # 3) Cross-attention: prompt queries image (get prompt-aware summarization)
        # we will pool K=some tokens to match P dimensions or produce prompt summary per patch
        # produce prompt->img target then broadcast back (we use prompt2img with prompt as Q and img as KV)
        prompt2img_out, attn_pi = self.prompt2img(prompt_ca, img_ca)  # [B, L, D]
        # pool prompt2img_out to per-patch representation via attention affinity:
        # compute affinity A: [B, P, L]
        A = torch.matmul(self.affine_i(img_ca), self.affine_p(prompt_ca).transpose(1,2)) / math.sqrt(D)
        A = torch.softmax(A, dim=-1)  # [B, P, L]
        ce = torch.matmul(A, prompt2img_out)  # [B, P, D]  correlation enhancement

        # 4) Build combination tensor per patch: [img, img_attn_out, ce, img * img_attn_out]
        combo = torch.cat([img_feats, img_attn_out, ce, img_feats * img_attn_out], dim=-1)  # [B, P, 4D]

        # 5) Fusion and reduce back to D
        fused = self.ffb(combo)  # [B, P, D]

        # 6) gating + residual
        # gating computed from global pooled info
        pooled_fused = fused.mean(dim=1)    # [B, D]
        pooled_img = img_feats.mean(dim=1)  # [B, D]
        g = self.gate(torch.cat([pooled_fused, pooled_img], dim=-1))  # [B, D]
        g = g.unsqueeze(1)  # [B,1,D]
        out = g * fused + (1 - g) * img_feats  # [B, P, D]

        return out
