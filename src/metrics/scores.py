import numpy as np
import torch
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist

FEATURE_BATCH = 100
IS_EPS = 1e-12
PR_K = 3

@torch.no_grad()
def get_features(images, encoder, batch_size = FEATURE_BATCH):
    all_feats = []
    all_probs = []
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]
        x = encoder.preprocess(chunk).to(encoder.device)
        feats, probs = encoder(x)
        all_probs.append(probs.cpu().numpy())
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0), np.concatenate(all_probs, axis=0)

def compute_fid(real_feats, fake_feats):
    mu_real = np.mean(real_feats, axis = 0)
    mu_fake = np.mean(fake_feats, axis = 0)
    sigma_real = np.cov(real_feats, rowvar=False)
    sigma_fake = np.cov(fake_feats, rowvar=False)
    mean_term = np.sum((mu_real - mu_fake) ** 2)
    covmean = sqrtm(sigma_real @ sigma_fake)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return mean_term + np.trace(sigma_real + sigma_fake - 2*covmean)


def compute_is(probs):
    p_y = np.mean(probs, axis = 0)
    log_p = np.log(probs + IS_EPS)
    log_py = np.log(p_y + IS_EPS)
    kl = np.sum(probs * (log_p - log_py), axis=1)
    return np.exp(np.mean(kl))

def knn_radii(feats, k):
    dists = cdist(feats, feats)
    sorted_dists = np.sort(dists, axis = 1)
    return sorted_dists[:, k]

def fraction_inside(query, ref, ref_radii):
    dists = cdist(query, ref)
    inside = dists<=ref_radii
    any_inside = np.any(inside, axis=1)
    return np.mean(any_inside)