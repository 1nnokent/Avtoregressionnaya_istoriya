import numpy as np

BATCH = 50
PIXEL_MIN = 0.0
PIXEL_MAX = 1.0

def generate_images(model, decode_fn, num_images, temperature, device, batch_size = BATCH, top_k=None):
    all_images = []
    for start in range(0, num_images, batch_size):
        n = min(batch_size, num_images - start)
        tokens = model.generate(num_samples = n, seq_len = model.config.max_seq_len, device = device, temperature = temperature, top_k = top_k)
        tokens_np = tokens.cpu().numpy()
        all_images.extend(np.clip(decode_fn(i), PIXEL_MIN, PIXEL_MAX) for i in tokens_np)

    return np.stack(all_images)