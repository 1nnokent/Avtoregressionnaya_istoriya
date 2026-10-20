"""Encoder-independent scores; float64 Fréchet computation handles singular covariances."""
import numpy as np
import torch
from scipy.spatial.distance import cdist

PR_K = 3


def _features(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected at least two finite feature vectors")
    return values


@torch.no_grad()
def get_features(images, encoder, batch_size=64):
    features, probabilities = [], []
    for start in range(0, len(images), batch_size):
        x = encoder.preprocess(images[start:start + batch_size]).to(encoder.device)
        feats, probs = encoder(x)
        features.append(feats.cpu().numpy())
        if probs is not None:
            probabilities.append(probs.cpu().numpy())
    return np.concatenate(features), np.concatenate(probabilities) if probabilities else None


def compute_fid(real_feats, fake_feats):
    """Fréchet feature distance. Only Inception features give the Inception variant."""
    real, fake = _features(real_feats), _features(fake_feats)
    if real.shape[1] != fake.shape[1]:
        raise ValueError("Feature dimensions must match")
    mean_term = np.sum((real.mean(0) - fake.mean(0)) ** 2)
    a = (real - real.mean(0)) / np.sqrt(len(real) - 1)
    b = (fake - fake.mean(0)) / np.sqrt(len(fake) - 1)
    if max(len(a), len(b)) <= a.shape[1]:
        cross_trace = np.linalg.svd(a @ b.T, compute_uv=False).sum()
    else:
        ca, cb = a.T @ a, b.T @ b
        values, vectors = np.linalg.eigh(ca)
        root = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
        product = root @ cb @ root
        cross_trace = np.sqrt(np.maximum(np.linalg.eigvalsh((product + product.T) / 2), 0)).sum()
    return float(max(0, mean_term + np.sum(a * a) + np.sum(b * b) - 2 * cross_trace))


def compute_is(probs):
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or len(probs) < 2 or not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError("Expected finite nonnegative class probabilities")
    if not np.allclose(probs.sum(1), 1, atol=1e-5):
        raise ValueError("Class probabilities must sum to one")
    marginal = probs.mean(0)
    kl = np.sum(probs * (np.log(np.maximum(probs, 1e-12)) - np.log(np.maximum(marginal, 1e-12))), axis=1)
    return float(np.exp(np.mean(kl)))


def knn_radii(feats, k, chunk_size=256):
    feats = _features(feats)
    if not isinstance(k, int) or not 1 <= k < len(feats):
        raise ValueError("k must be an integer in [1, number of features - 1]")
    radii = []
    for start in range(0, len(feats), chunk_size):
        distances = cdist(feats[start:start + chunk_size], feats)
        # The zero distance to the point itself is index zero; kth neighbor is index k.
        radii.append(np.partition(distances, k, axis=1)[:, k])
    return np.concatenate(radii)


def fraction_inside(query, ref, ref_radii, chunk_size=256):
    query, ref = _features(query), _features(ref)
    ref_radii = np.asarray(ref_radii)
    if ref_radii.shape != (len(ref),) or np.any(ref_radii < 0):
        raise ValueError("Expected one nonnegative radius per reference feature")
    count = 0
    for start in range(0, len(query), chunk_size):
        count += np.any(cdist(query[start:start + chunk_size], ref) <= ref_radii, axis=1).sum()
    return float(count / len(query))
