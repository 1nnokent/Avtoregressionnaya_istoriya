import numpy as np


def generate_images(model, decode_fn, num_images, temperature, device, batch_size=64, top_k=None):
    images = []
    for start in range(0, num_images, batch_size):
        tokens = model.generate(num_samples=min(batch_size, num_images - start),
                                seq_len=model.config.max_seq_len, device=device,
                                temperature=temperature, top_k=top_k)
        images.append(decode_fn(tokens.cpu().numpy()))
    return np.concatenate(images)
