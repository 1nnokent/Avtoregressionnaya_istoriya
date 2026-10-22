import math
from pathlib import Path

import torch
import torch.nn.functional as F
PRECISIONS = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def get_tokens(batch):
    if isinstance(batch, torch.Tensor):
        tokens = batch
    elif isinstance(batch, dict):
        tokens = batch["tokens"]
    elif isinstance(batch, (tuple, list)):
        tokens = batch[0]
    else:
        raise TypeError("Unsupported batch format")

    if not isinstance(tokens, torch.Tensor):
        raise TypeError("Batch tokens must be a tensor")

    if tokens.ndim != 2:
        raise ValueError("Expected tokens with shape [batch, sequence]")

    return tokens.long()


def prepare_batch(batch, model, device):
    tokens = get_tokens(batch).to(device)

    if tokens.size(0) == 0 or tokens.size(1) == 0:
        raise ValueError("Batch and sequence must not be empty")

    if tokens.size(1) > model.config.max_seq_len:
        raise ValueError("Sequence exceeds max_seq_len")

    if torch.any(tokens < 0) or torch.any(
        tokens >= model.config.vocab_size
    ):
        raise ValueError("Dataset token ID is outside the vocabulary")

    bos = torch.full(
        (tokens.size(0), 1),
        model.bos_token_id,
        dtype=torch.long,
        device=device,
    )

    inputs = torch.cat((bos, tokens[:, :-1]), dim=1)

    return inputs, tokens


@torch.no_grad()
def evaluate(
    model,
    val_loader,
    device,
    max_batches=None,
    amp_dtype=None,
):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    try:
        for batch_idx, batch in enumerate(val_loader):
            if (
                max_batches is not None
                and batch_idx >= max_batches
            ):
                break

            inputs, targets = prepare_batch(
                batch,
                model,
                device,
            )

            with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                logits = model(inputs)

            loss = F.cross_entropy(
                logits.float().reshape(-1, model.config.vocab_size),
                targets.reshape(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += targets.numel()

    finally:
        model.train(was_training)

    if total_tokens == 0:
        raise ValueError("Validation loader is empty")

    mean_loss = total_loss / total_tokens

    return {
        "val_loss": mean_loss,
        "val_perplexity": math.exp(min(mean_loss, 80.0)),
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    step,
    epoch,
    best_val_loss,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_config": vars(model.config).copy(),
        },
        path,
    )


def load_checkpoint(
    path,
    model,
    optimizer=None,
    device="cpu",
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    if checkpoint["model_config"] != vars(model.config):
        raise ValueError(
            "Checkpoint model configuration does not match"
        )

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def train_model(
    model,
    train_loader,
    val_loader,
    config,
    device=None,
    on_step=None,
):
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    amp_dtype = PRECISIONS[config.get("precision", "fp32")]
    use_amp = amp_dtype is not None
    scaler = torch.amp.GradScaler(device.type, enabled=(amp_dtype == torch.float16))

    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    resume_from = config.get("resume_from")

    if resume_from is not None:
        checkpoint = load_checkpoint(
            resume_from,
            model,
            optimizer,
            device,
        )

        step = checkpoint["step"]
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

        print(
            f"Resuming from step={step}, "
            f"next_epoch={start_epoch + 1}"
        )

    model.train()

    max_steps = config.get("max_steps")

    if max_steps is not None and step >= max_steps:
        return model

    for epoch in range(start_epoch, config["epochs"]):
        model.train()

        for batch in train_loader:
            inputs, targets = prepare_batch(
                batch,
                model,
                device,
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, model.config.vocab_size),
                    targets.reshape(-1),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            step += 1

            if step % config["log_interval"] == 0:
                print(
                    f"epoch={epoch + 1} "
                    f"step={step} "
                    f"train_loss={loss.item():.4f}"
                )

            if step % config["eval_interval"] == 0:
                metrics = evaluate(
                    model,
                    val_loader,
                    device,
                    config.get("max_val_batches"),
                    amp_dtype,
                )

                print(
                    f"step={step} "
                    f"val_loss={metrics['val_loss']:.4f} "
                    f"val_perplexity="
                    f"{metrics['val_perplexity']:.2f}"
                )

                if metrics["val_loss"] < best_val_loss:
                    best_val_loss = metrics["val_loss"]

                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        model,
                        optimizer,
                        step,
                        epoch,
                        best_val_loss,
                    )

            if on_step is not None:
                on_step(model, step)

            if step % config["checkpoint_interval"] == 0:
                save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    model,
                    optimizer,
                    step,
                    epoch,
                    best_val_loss,
                )

            if max_steps is not None and step >= max_steps:
                break

        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model,
            optimizer,
            step,
            epoch,
            best_val_loss,
        )

        if max_steps is not None and step >= max_steps:
            break

    return model
