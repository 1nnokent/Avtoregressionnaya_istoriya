from numbers import Integral

import numpy as np
from scipy.spatial.distance import cdist

FEATURE_BATCH = 100
DIST_CHUNK = 1024
IS_EPS = 1e-12
PR_K = 3
SELF_NEIGHBOR = 0


def reconstruction_metrics(original, reconstructed, data_range=1.0):
    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be finite and positive")
    if original.shape != reconstructed.shape:
        raise ValueError("original and reconstructed images must have matching shapes")
    if original.size == 0:
        raise ValueError("images must be non-empty")
    if not np.isfinite(original).all() or not np.isfinite(reconstructed).all():
        raise ValueError("images must contain only finite values")

    difference = original.astype(np.float64) - reconstructed
    mse = float(np.mean(difference**2))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(data_range**2 / mse))
    return {"mse": mse, "psnr": psnr}


def token_usage_metrics(tokens, vocab_size):
    tokens = np.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError("tokens must be two-dimensional")
    if tokens.size == 0:
        raise ValueError("tokens must be non-empty")
    if not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError("tokens must have an integer dtype")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, Integral):
        raise TypeError("vocab_size must be an integer")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if tokens.min() < 0 or tokens.max() >= vocab_size:
        raise ValueError("token ID is outside the vocabulary")

    counts = np.bincount(tokens.ravel(), minlength=vocab_size)
    frequencies = counts / counts.sum()
    used = frequencies > 0
    entropy = float(-np.sum(frequencies[used] * np.log2(frequencies[used])))
    used_tokens = int(np.count_nonzero(used))
    return {
        "sequence_length": int(tokens.shape[1]),
        "vocab_size": int(vocab_size),
        "token_count": int(tokens.size),
        "used_tokens": used_tokens,
        "unused_tokens": int(vocab_size - used_tokens),
        "utilization": float(used_tokens / vocab_size),
        "entropy_bits": entropy,
        "effective_vocab_size": float(2**entropy),
        "frequencies": frequencies.tolist(),
    }


def get_features(images, encoder, batch_size=FEATURE_BATCH):
    feats, probs = [], []
    for start in range(0, len(images), batch_size):
        f, p = encoder(encoder.preprocess(images[start:start + batch_size]))
        feats.append(f.cpu().numpy())
        probs.append(p.cpu().numpy())
    return np.concatenate(feats), np.concatenate(probs)


def _gaussian_stats(feats):
    return feats.mean(axis=0), np.cov(feats, rowvar=False)


def _trace_sqrt_product(sigma_a, sigma_b):
    eigvals = np.linalg.eigvals(sigma_a @ sigma_b).real
    return np.sqrt(np.clip(eigvals, 0, None)).sum()


def compute_fid(real_feats, fake_feats):
    mu_r, sigma_r = _gaussian_stats(real_feats)
    mu_f, sigma_f = _gaussian_stats(fake_feats)
    mean_term = np.sum((mu_r - mu_f) ** 2)
    cov_term = np.trace(sigma_r) + np.trace(sigma_f) - 2 * _trace_sqrt_product(sigma_r, sigma_f)
    return float(mean_term + cov_term)


def compute_is(probs):
    p_y = probs.mean(axis=0)
    kl = np.sum(probs * (np.log(probs + IS_EPS) - np.log(p_y + IS_EPS)), axis=1)
    return float(np.exp(kl.mean()))


def knn_radii(feats, k=PR_K, chunk=DIST_CHUNK):
    assert k > SELF_NEIGHBOR, 'k-th neighbor must exclude the point itself'
    radii = np.empty(len(feats))
    for start in range(0, len(feats), chunk):
        dists = cdist(feats[start:start + chunk], feats)
        radii[start:start + chunk] = np.partition(dists, k, axis=1)[:, k]
    return radii


def fraction_inside(query, ref, ref_radii, chunk=DIST_CHUNK):
    inside = np.empty(len(query), dtype=bool)
    for start in range(0, len(query), chunk):
        dists = cdist(query[start:start + chunk], ref)
        inside[start:start + chunk] = np.any(dists <= ref_radii, axis=1)
    return float(inside.mean())


def compute_precision_recall(real_feats, fake_feats, k=PR_K):
    precision = fraction_inside(fake_feats, real_feats, knn_radii(real_feats, k))
    recall = fraction_inside(real_feats, fake_feats, knn_radii(fake_feats, k))
    return precision, recall
