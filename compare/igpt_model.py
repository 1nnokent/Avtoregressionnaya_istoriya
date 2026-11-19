import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        inv = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * inv).to(x.dtype) * self.weight

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head, query_chunk=128):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.query_chunk = query_chunk
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, past=None, use_cache=False):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # [B, heads, T, head_dim]
        if past is not None:
            pk, pv = past
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        # Для T4 делим 4096-token attention на блоки запросов.
        # Каждый запрос на позиции i видит строго ключи 0..i (полная causal attention).
        # Это НЕ локальный attention и НЕ усечение контекста.
        if past is None and T > self.query_chunk:
            chunks = []
            for start in range(0, T, self.query_chunk):
                end = min(start + self.query_chunk, T)
                # Ключи после end не нужны ни одному запросу в этом блоке.
                qi = q[:, :, start:end, :]
                ki, vi = k[:, :, :end, :], v[:, :, :end, :]
                # В SDPA True означает РАЗРЕШЁННУЮ позицию.
                qpos = torch.arange(start, end, device=x.device)[:, None]
                kpos = torch.arange(end, device=x.device)[None, :]
                causal_mask = (qpos >= kpos)
                yi = F.scaled_dot_product_attention(
                    qi, ki, vi, attn_mask=causal_mask, dropout_p=0.0,
                    is_causal=False
                )
                chunks.append(yi)
            y = torch.cat(chunks, dim=2)
        else:
            # Короткие последовательности и по-токенная генерация с KV-cache.
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=(past is None and T > 1), dropout_p=0.0
            )
        y = y.transpose(1, 2).contiguous().reshape(B, T, D)
        out = self.proj(y)
        return (out, (k, v)) if use_cache else out

class Block(nn.Module):
    def __init__(self, dim, n_head, query_chunk=128):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_head, query_chunk=query_chunk)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=False),
            QuickGELU(),
            nn.Linear(4 * dim, dim, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def forward_cached(self, x, past):
        attn_out, present = self.attn(self.norm1(x), past=past, use_cache=True)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, present

class ImageGPTSmall(nn.Module):
    def __init__(self, n_layer=24, n_embd=512, n_head=8, n_vocab=512,
                 seq_len=4096, query_chunk=128):
        super().__init__()
        self.seq_len = seq_len
        self.n_embd = n_embd
        self.wte = nn.Embedding(n_vocab, n_embd)
        self.sos = nn.Parameter(torch.empty(n_embd))
        self.wpe = nn.Parameter(torch.empty(seq_len, n_embd))
        self.blocks = nn.ModuleList([Block(n_embd, n_head, query_chunk) for _ in range(n_layer)])
        self.norm_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, n_vocab, bias=False)  # untied, как в original iGPT
        self.apply(self._init_linear)
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.normal_(self.sos, std=0.02)
        nn.init.normal_(self.wpe, std=0.01)
        nn.init.zeros_(self.lm_head.weight)

    @staticmethod
    def _init_linear(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, tokens, gradient_checkpointing=False):
        B, T = tokens.shape
        assert 0 < T <= self.seq_len
        # Логиты на позиции i предсказывают token[i]: [SOS, x0, ..., x(T-2)]
        sos = self.sos.view(1, 1, -1).expand(B, 1, -1)
        x = torch.cat([sos, self.wte(tokens[:, :-1])], dim=1)
        x = x + self.wpe[:T].unsqueeze(0)
        for block in self.blocks:
            if gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        logits = self.lm_head(self.norm_f(x))
        return logits

    @torch.inference_mode()
    def generate(self, n_samples=1, temperature=1.0, top_k=128, prefix=None):
        """Сэмплинг всех 4096 токенов с KV-cache (или продолжение префикса)."""
        assert temperature > 0
        self.eval()
        device = next(self.parameters()).device
        if prefix is not None:
            prefix = torch.as_tensor(prefix, device=device, dtype=torch.long)
            if prefix.ndim == 1:
                prefix = prefix.unsqueeze(0)
            assert prefix.shape[1] <= self.seq_len
            n_samples = prefix.shape[0]
        cache = [None] * len(self.blocks)
        generated = torch.empty((n_samples, self.seq_len), device=device, dtype=torch.long)
        prev_token = None
        for i in tqdm(range(self.seq_len), desc='Generating 64x64 image'):
            if i == 0:
                x = self.sos.view(1, 1, -1).expand(n_samples, 1, -1)
            else:
                x = self.wte(prev_token.unsqueeze(1))
            x = x + self.wpe[i:i+1].unsqueeze(0)
            for j, block in enumerate(self.blocks):
                x, cache[j] = block.forward_cached(x, cache[j])
            logits = self.lm_head(self.norm_f(x))[:, -1, :].float() / temperature
            if prefix is not None and i < prefix.shape[1]:
                next_token = prefix[:, i]
            else:
                if top_k is not None and top_k < logits.shape[-1]:
                    threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float('-inf'))
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(1)
            generated[:, i] = next_token
            prev_token = next_token
        return generated
