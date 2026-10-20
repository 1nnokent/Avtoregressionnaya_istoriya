import math
from pathlib import Path

import torch
import torch.nn.functional as F


def get_tokens(batch):
    tokens = batch if isinstance(batch, torch.Tensor) else batch["tokens"] if isinstance(batch, dict) else batch[0]
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.dtype != torch.long:
        raise ValueError("Expected int64 tokens [batch, sequence]")
    return tokens


def prepare_batch(batch, model, device):
    tokens = get_tokens(batch).to(device)
    if tokens.numel() == 0 or tokens.size(1) > model.config.max_seq_len:
        raise ValueError("Empty batch or sequence exceeds max_seq_len")
    if torch.any(tokens < 0) or torch.any(tokens >= model.config.vocab_size):
        raise ValueError("Dataset token ID is outside the vocabulary")
    bos = torch.full((tokens.size(0), 1), model.bos_token_id, dtype=torch.long, device=device)
    return torch.cat((bos, tokens[:, :-1]), dim=1), tokens


@torch.no_grad()
def evaluate(model, loader, device):
    was_training = model.training
    model.eval()
    loss_sum, count, correct = 0.0, 0, 0
    try:
        for batch in loader:
            inputs, targets = prepare_batch(batch, model, device)
            logits = model(inputs)
            loss_sum += F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum").item()
            correct += (logits.argmax(-1) == targets).sum().item()
            count += targets.numel()
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("Evaluation loader is empty")
    nll = loss_sum / count
    return {"nll": nll, "bits_per_token": nll / math.log(2),
            "perplexity": math.exp(min(nll, 80)), "token_accuracy": correct / count}


def save_checkpoint(path, model, optimizer, step, epoch, best_val_loss):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "epoch": epoch, "best_val_loss": best_val_loss, "model_config": vars(model.config)}, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["model_config"] != vars(model.config):
        raise ValueError("Checkpoint model configuration does not match")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def train_model(model, train_loader, val_loader, config, device, output_dir, logger):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    output_dir = Path(output_dir)
    initial = evaluate(model, val_loader, device)
    logger.log({f"val_initial/{k}": v for k, v in initial.items()}, 0)
    best, step, last_eval, epoch = initial["nll"], 0, -1, -1
    save_checkpoint(output_dir / "best.pt", model, optimizer, step, epoch, best)

    def validate():
        nonlocal best, last_eval
        metrics = evaluate(model, val_loader, device)
        logger.log({f"val/{k}": v for k, v in metrics.items()}, step)
        last_eval = step
        if metrics["nll"] < best:
            best = metrics["nll"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, step, epoch, best)

    running_loss, running_tokens = 0.0, 0
    for epoch in range(config["epochs"]):
        model.train()
        for batch in train_loader:
            inputs, targets = prepare_batch(batch, model, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            step += 1
            running_loss += loss.item() * targets.numel()
            running_tokens += targets.numel()
            if step % config["log_interval"] == 0:
                logger.log({"train/nll": running_loss / running_tokens}, step)
                running_loss, running_tokens = 0.0, 0
            if step % config["eval_interval"] == 0:
                validate()
            if config["max_steps"] is not None and step >= config["max_steps"]:
                break
        save_checkpoint(output_dir / "latest.pt", model, optimizer, step, epoch, best)
        if config["max_steps"] is not None and step >= config["max_steps"]:
            break
    if last_eval != step:
        validate()
    save_checkpoint(output_dir / "latest.pt", model, optimizer, step, epoch, best)
    return {"steps": step, "epochs_started": epoch + 1, "initial_val": initial, "best_val_nll": best}
