import json
from functools import partial
import torch

from src.model import ImageTokenTransformer, ModelConfig
from src.pipeline import BaselineParams, tokens_to_image
from src.datasets_loader import load_normed_mnist_images
from src.generation import generate_images
from src.metrics.encoders import ENCODERS
from src.metrics.scores import get_features, compute_fid, compute_is, knn_radii, fraction_inside, PR_K

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'checkpoints/best.pt'
PARAMS_PATH = 'checkpoints/params.json'
NUM_EVAL_SAMPLES = 200
TEMPERATURE = 0.7
SEED = 0

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
model = ImageTokenTransformer(ModelConfig(**checkpoint['model_config'])).to(DEVICE)
model.load_state_dict(checkpoint['model'])
model.eval()

with open(PARAMS_PATH) as f:
    params = BaselineParams(**json.load(f))
decode_fn = partial(tokens_to_image, params=params)

test_images = load_normed_mnist_images(train=False)
real_images = test_images[:NUM_EVAL_SAMPLES]

torch.manual_seed(SEED)
fake_images = generate_images(model, decode_fn, NUM_EVAL_SAMPLES, TEMPERATURE, DEVICE)

for name, encoder_cls in ENCODERS.items():
    encoder = encoder_cls(DEVICE)
    real_feats, real_probs = get_features(real_images, encoder)
    fake_feats, fake_probs = get_features(fake_images, encoder)
    fid = compute_fid(real_feats, fake_feats)
    is_fake = compute_is(fake_probs)
    is_real = compute_is(real_probs)
    precision = fraction_inside(fake_feats, real_feats, knn_radii(real_feats, PR_K))
    recall = fraction_inside(real_feats, fake_feats, knn_radii(fake_feats, PR_K))
    print(f'{name}: FID {fid:.2f}  IS fake {is_fake:.2f} / real {is_real:.2f}  '
          f'P {precision:.3f}  R {recall:.3f}')