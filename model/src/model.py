import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelConfig:
    def __init__(
        self,
        vocab_size=256,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=784,
        dropout=0.1,
    ):
        if min(
            vocab_size,
            d_model,
            n_heads,
            n_layers,
            d_ff,
            max_seq_len,
        ) <= 0:
            raise ValueError("All dimensions must be positive")

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if (d_model // n_heads) % 2 != 0:
            raise ValueError("Head dimension must be even for RoPE")

        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout


class RoPE(nn.Module):
    def __init__(self, head_dim, theta=10000.0):
        super().__init__()

        inv_freq = 1.0 / (
            theta ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / head_dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

    def forward(self, x):
        seq_len = x.size(-2)

        positions = torch.arange(
            seq_len,
            device=x.device,
            dtype=self.inv_freq.dtype,
        )

        freqs = torch.outer(positions, self.inv_freq)

        cos = freqs.cos().to(x.dtype)[None, None, :, :]
        sin = freqs.sin().to(x.dtype)[None, None, :, :]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated = torch.stack(
            (
                x_even * cos - x_odd * sin,
                x_even * sin + x_odd * cos,
            ),
            dim=-1,
        )

        return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(
            config.d_model,
            3 * config.d_model,
        )

        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model,
        )

        self.rope = RoPE(self.head_dim)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        qkv = self.qkv(x).view(
            batch_size,
            seq_len,
            3,
            self.n_heads,
            self.head_dim,
        )

        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        q = self.rope(q)
        k = self.rope(k)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=(
                self.attn_dropout if self.training else 0.0
            ),
        )

        y = y.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            d_model,
        )

        return self.resid_dropout(self.out_proj(y))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ImageTokenTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.bos_token_id = config.vocab_size

        self.token_embedding = nn.Embedding(
            config.vocab_size + 1,
            config.d_model,
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        if tokens.ndim != 2:
            raise ValueError(
                "tokens must have shape [batch, sequence]"
            )

        if tokens.size(1) == 0:
            raise ValueError("Sequence must not be empty")

        if tokens.size(1) > self.config.max_seq_len:
            raise ValueError("Sequence exceeds max_seq_len")

        if tokens.dtype != torch.long:
            raise TypeError("tokens must have dtype torch.long")

        if torch.any(tokens < 0) or torch.any(
            tokens > self.bos_token_id
        ):
            raise ValueError("Token ID is out of range")

        x = self.token_embedding(tokens)
        x = self.blocks(x)
        x = self.final_norm(x)

        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        num_samples,
        seq_len,
        device=None,
        temperature=1.0,
        top_k=None,
    ):
        if num_samples < 1:
            raise ValueError("num_samples must be positive")

        if not 1 <= seq_len <= self.config.max_seq_len:
            raise ValueError("Invalid seq_len")

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")

        if device is None:
            device = next(self.parameters()).device

        was_training = self.training
        self.eval()

        try:
            tokens = torch.full(
                (num_samples, 1),
                self.bos_token_id,
                dtype=torch.long,
                device=device,
            )

            for _ in range(seq_len):
                logits = self(tokens)[:, -1, :]
                logits = logits / temperature

                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    threshold = torch.topk(
                        logits,
                        k,
                        dim=-1,
                    ).values[:, -1:]

                    logits = logits.masked_fill(
                        logits < threshold,
                        float("-inf"),
                    )

                probs = F.softmax(logits, dim=-1)

                next_token = torch.multinomial(
                    probs,
                    num_samples=1,
                )

                tokens = torch.cat(
                    (tokens, next_token),
                    dim=1,
                )

            return tokens[:, 1:]

        finally:
            self.train(was_training)
