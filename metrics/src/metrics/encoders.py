import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from transformers import CLIPModel, CLIPTokenizer


INCEPTION_SIZE = 299
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLIP_SIZE = 224
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'
NUM_DIGITS = 10
DIGIT_PROMPTS = [f'a handwritten digit {d}' for d in range(NUM_DIGITS)]


def preprocess(images, size, mean, std):
  x = torch.tensor(np.asarray(images), dtype=torch.float32)
  x = x.unsqueeze(1)
  x = x.repeat(1, 3, 1, 1)
  x = F.interpolate(x, size = size, mode = 'bilinear', align_corners=False)
  mean_n = torch.tensor(mean).view(1, 3, 1, 1)
  std_n = torch.tensor(std).view(1, 3, 1, 1)
  x = (x - mean_n) / std_n
  return x

class InceptionEncoder:
    def __init__(self, device):
        self.device = device
        self.net = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.fc = self.net.fc.eval().to(device)
        self.net.fc = nn.Identity()
        self.net = self.net.eval().to(device)

    def preprocess(self, images):
        return preprocess(images, INCEPTION_SIZE, IMAGENET_MEAN, IMAGENET_STD)

    @torch.no_grad()
    def __call__(self, x):
        feats = self.net(x)
        probs = F.softmax(self.fc(feats), dim=-1)
        return feats, probs

class CLIPEncoder:
    def __init__(self, device, class_prompts=DIGIT_PROMPTS):
        self.device = device
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).eval().to(device)
        clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
        text_inputs = clip_tokenizer(class_prompts, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            text_feats = self.model.get_text_features(**text_inputs)
        self.text_feats = F.normalize(text_feats, dim=-1)

    def preprocess(self, images):
        return preprocess(images, CLIP_SIZE, CLIP_MEAN, CLIP_STD)

    @torch.no_grad()
    def __call__(self, x):
        feats = self.model.get_image_features(pixel_values=x)
        feats_n = F.normalize(feats, dim=-1)
        logits = self.model.logit_scale.exp() * feats_n @ self.text_feats.T
        probs = F.softmax(logits, dim=-1)
        return feats, probs

ENCODERS = {
    'inception': InceptionEncoder,
    'clip': CLIPEncoder,
}