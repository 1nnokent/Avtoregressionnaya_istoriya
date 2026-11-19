import torch
from torch import nn
from torch.nn import functional as F

from compare.igpt_model import ImageGPTSmall

FFN_MULTIPLIER = 4
CUDA_CACHE_DTYPE = torch.float16


class IGPTConfig:
    def __init__(
        self,
        vocab_size=256,
        max_seq_len=3072,
        d_model=512,
        n_heads=8,
        n_layers=24,
        d_ff=None,
        dropout=0.0,
        query_chunk=4096,
        gradient_checkpointing=True,
    ):
        if min(vocab_size, max_seq_len, d_model, n_heads, n_layers, query_chunk) <= 0:
            raise ValueError("All dimensions must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if d_ff is not None and d_ff != FFN_MULTIPLIER * d_model:
            raise ValueError(f"iGPT uses d_ff = {FFN_MULTIPLIER} * d_model")
        if dropout != 0:
            raise ValueError("iGPT has no dropout; set model.params.dropout: 0.0")

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = FFN_MULTIPLIER * d_model
        self.dropout = 0.0
        self.query_chunk = query_chunk
        self.gradient_checkpointing = bool(gradient_checkpointing)


class SpectralImageGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bos_token_id = config.vocab_size   # BOS из prepare_batch; внутри заменяется на SOS iGPT
        self.igpt = ImageGPTSmall(
            n_layer=config.n_layers,
            n_embd=config.d_model,
            n_head=config.n_heads,
            n_vocab=config.vocab_size,
            seq_len=config.max_seq_len,
            query_chunk=config.query_chunk,
        )

    def forward(self, inputs):
        if inputs.ndim != 2 or inputs.size(1) == 0:
            raise ValueError("inputs must have shape [batch, sequence]")
        if inputs.size(1) > self.config.max_seq_len:
            raise ValueError("Sequence exceeds max_seq_len")
        if inputs.dtype != torch.long:
            raise TypeError("inputs must have dtype torch.long")
        if torch.any(inputs[:, 0] != self.bos_token_id):
            raise ValueError("inputs must start with bos_token_id")
        rest = inputs[:, 1:]
        if torch.any(rest < 0) or torch.any(rest >= self.config.vocab_size):
            raise ValueError("Token ID is out of range")

        tokens = torch.cat((rest, rest.new_zeros(rest.size(0), 1)), dim=1)
        return self.igpt(tokens, gradient_checkpointing=self.config.gradient_checkpointing)

    def _decode_step(self, keys, values, position, prev_tokens, batch_size):
        igpt = self.igpt
        if position == 0:
            x = igpt.sos.view(1, 1, -1).expand(batch_size, 1, -1)
        else:
            x = igpt.wte(prev_tokens[:, None])
        x = x + igpt.wpe[position:position + 1][None]

        for layer, block in enumerate(igpt.blocks):
            attn = block.attn
            qkv = attn.qkv(block.norm1(x)).reshape(batch_size, 1, 3, attn.n_head, attn.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)     # [batch, heads, 1, head_dim]
            keys[layer, :, :, position:position + 1] = k
            values[layer, :, :, position:position + 1] = v
            y = F.scaled_dot_product_attention(                # новый токен видит всё прошлое
                q.to(keys.dtype),
                keys[layer, :, :, :position + 1],
                values[layer, :, :, :position + 1],
            )
            y = y.transpose(1, 2).reshape(batch_size, 1, -1).to(x.dtype)
            x = x + attn.proj(y)
            x = x + block.mlp(block.norm2(x))

        return igpt.lm_head(igpt.norm_f(x))[:, -1, :].float()

    @torch.no_grad()
    def generate(self, num_samples, seq_len, device=None, temperature=1.0, top_k=None):
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
        device = torch.device(device)
        on_cuda = device.type == "cuda"
        cache_dtype = CUDA_CACHE_DTYPE if on_cuda else torch.float32

        was_training = self.training
        self.eval()
        try:
            first_attn = self.igpt.blocks[0].attn
            shape = (len(self.igpt.blocks), num_samples, first_attn.n_head, seq_len, first_attn.head_dim)
            keys = torch.empty(shape, device=device, dtype=cache_dtype)
            values = torch.empty_like(keys)
            generated = torch.empty((num_samples, seq_len), dtype=torch.long, device=device)
            prev = None
            with torch.autocast(device_type="cuda", dtype=CUDA_CACHE_DTYPE, enabled=on_cuda):
                for position in range(seq_len):
                    logits = self._decode_step(keys, values, position, prev, num_samples) / temperature
                    if top_k is not None and top_k < logits.size(-1):
                        threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                        logits = logits.masked_fill(logits < threshold, float("-inf"))
                    prev = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(1)
                    generated[:, position] = prev
            return generated
        finally:
            self.train(was_training)

    @torch.no_grad()
    def max_cache_error(self, tokens):
        batch_size, length = tokens.shape
        bos = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=tokens.device)
        expected = self(torch.cat((bos, tokens[:, :-1]), dim=1)).float()
        first_attn = self.igpt.blocks[0].attn
        shape = (len(self.igpt.blocks), batch_size, first_attn.n_head, length, first_attn.head_dim)
        keys = torch.empty(shape, device=tokens.device)
        values = torch.empty_like(keys)
        steps = [
            self._decode_step(keys, values, position, tokens[:, position - 1] if position else None, batch_size)
            for position in range(length)
        ]
        return (torch.stack(steps, dim=1) - expected).abs().max().item()
