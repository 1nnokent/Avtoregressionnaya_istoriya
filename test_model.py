import torch
import torch.nn.functional as F

from src.model import ImageTokenTransformer, ModelConfig


def main():
    config = ModelConfig(
        vocab_size=256,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=784,
        dropout=0.1,
    )

    model = ImageTokenTransformer(config)

    batch_size = 4
    seq_len = 784

    tokens = torch.randint(
        0,
        config.vocab_size,
        (batch_size, seq_len),
    )

    bos = torch.full(
        (batch_size, 1),
        model.bos_token_id,
        dtype=torch.long,
    )

    inputs = torch.cat((bos, tokens[:, :-1]), dim=1)

    logits = model(inputs)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)

    loss = F.cross_entropy(
        logits.reshape(-1, config.vocab_size),
        tokens.reshape(-1),
    )

    loss.backward()

    generated = model.generate(
        num_samples=2,
        seq_len=32,
        top_k=40,
    )

    assert generated.shape == (2, 32)
    assert generated.min() >= 0
    assert generated.max() < config.vocab_size

    print("Forward pass:", logits.shape)
    print("Loss:", loss.item())
    print("Generation:", generated.shape)
    print("All checks passed")


if __name__ == "__main__":
    main()
