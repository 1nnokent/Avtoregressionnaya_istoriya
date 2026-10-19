MODEL_CONFIG = {
    "vocab_size": 256,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 4,
    "d_ff": 512,
    "max_seq_len": 784,
    "dropout": 0.1,
}

TRAIN_CONFIG = {
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "epochs": 20,
    "max_steps": 10000,
    "eval_interval": 200,
    "checkpoint_interval": 500,
    "log_interval": 20,
    "grad_clip": 1.0,
    "checkpoint_dir": "checkpoints/mnist_baseline",
    "max_val_batches": None,
    "resume_from": None,
}
