import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class FullSelfAttention(nn.Module):
    """Exact bidirectional SDPA over ALL keys; split only the query dimension."""
    def __init__(self, dim, n_head, query_chunk=128):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.d_head = dim // n_head
        self.query_chunk = query_chunk
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_head, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # Every query chunk attends to K/V across all 4096 positions.
        # No causal or local/window mask. T4 memory footprint is controlled
        # by the query chunk even if PyTorch uses the math SDPA backend.
        if N <= self.query_chunk:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0,
                                               is_causal=False)
        else:
            outs = []
            for lo in range(0, N, self.query_chunk):
                outs.append(F.scaled_dot_product_attention(
                    q[:, :, lo:lo + self.query_chunk, :], k, v,
                    dropout_p=0.0, is_causal=False))
            y = torch.cat(outs, dim=2)
        return self.proj(y.transpose(1, 2).contiguous().view(B, N, D))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, query_chunk=128):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = FullSelfAttention(dim, heads, query_chunk)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio, bias=False),
            QuickGELU(),
            nn.Linear(dim * mlp_ratio, dim, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def sinusoidal_timestep_embedding(timesteps, dim):
    """Return sinusoidal [B, dim], for integer t in 0..999."""
    half = dim // 2
    scale = math.log(10000) / max(half - 1, 1)
    frequencies = torch.exp(-torch.arange(half, device=timesteps.device,
                                          dtype=torch.float32) * scale)
    phase = timesteps.float()[:, None] * frequencies[None, :]
    out = torch.cat([phase.cos(), phase.sin()], dim=1)
    if dim % 2:
        out = F.pad(out, (0, 1))
    return out


class IGPTRGBDenoiser(nn.Module):
    def __init__(self, n_layer=24, n_embd=512, n_head=8, seq_len=4096,
                 mlp_ratio=4, query_chunk=128):
        super().__init__()
        self.seq_len = seq_len
        self.dim = n_embd
        self.rgb_in = nn.Linear(3, n_embd, bias=True)
        self.wpe = nn.Parameter(torch.empty(seq_len, n_embd))
        # timestep embedding is ADDED ONCE at the input (no adaLN, no gates)
        self.time_mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd), nn.SiLU(), nn.Linear(n_embd, n_embd))
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, mlp_ratio, query_chunk)
            for _ in range(n_layer)
        ])
        self.norm_f = RMSNorm(n_embd)
        self.rgb_out = nn.Linear(n_embd, 3, bias=True)
        self.apply(self._init_linear)
        nn.init.normal_(self.wpe, std=0.01)
        # Same zero-initialized final head convention as previous iGPT notebook.
        nn.init.zeros_(self.rgb_out.weight)
        nn.init.zeros_(self.rgb_out.bias)

    @staticmethod
    def _init_linear(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x_t, t, gradient_checkpointing=False):
        # x_t: [B, 4096, 3] floats (no quantization)
        B, N, C = x_t.shape
        assert C == 3 and N <= self.seq_len
        assert t.shape == (B,)
        t_embed = self.time_mlp(sinusoidal_timestep_embedding(t, self.dim))
        h = self.rgb_in(x_t) + self.wpe[:N][None, :, :] + t_embed[:, None, :]
        for block in self.blocks:
            if gradient_checkpointing and self.training:
                h = checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        return self.rgb_out(self.norm_f(h))  # predicted epsilon, [B,N,3]
