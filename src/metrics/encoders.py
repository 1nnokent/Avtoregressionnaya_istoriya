"""Lazy optional encoders and a local, train-split-only MNIST reference classifier."""
import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def preprocess(images, size, mean, std):
    x = torch.as_tensor(np.asarray(images), dtype=torch.float32)
    if x.ndim == 3:
        x = x[:, None]
    if x.ndim != 4 or x.shape[1] not in (1, 3):
        raise ValueError("Expected NCHW grayscale or RGB images")
    if len(mean) == 3 and x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    if len(mean) == 1 and x.shape[1] != 1:
        raise ValueError("MNIST encoder expects grayscale images")
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return (x - torch.tensor(mean)[None, :, None, None]) / torch.tensor(std)[None, :, None, None]


class PixelEncoder:
    def __init__(self, device):
        self.device = device

    def preprocess(self, images):
        x = torch.as_tensor(images, dtype=torch.float32)
        return F.adaptive_avg_pool2d(x, (8, 8)).flatten(1)

    def __call__(self, x):
        return x, None


class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(512, 64), nn.ReLU())
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        features = self.features(x)
        return features, self.head(features)


class MNISTEncoder:
    def __init__(self, device, checkpoint):
        self.device = device
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        self.net = DigitNet().to(device)
        self.net.load_state_dict(state["model"])
        self.net.eval()
        self.val_accuracy = state["val_accuracy"]

    def preprocess(self, images):
        return preprocess(images, 28, [0.1307], [0.3081])

    @torch.no_grad()
    def __call__(self, x):
        features, logits = self.net(x)
        return features, logits.softmax(-1)


def train_mnist_encoder(train_data, val_data, device, path, epochs, seed, logger):
    torch.manual_seed(seed)
    net = DigitNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    train_images, train_labels = train_data
    val_images, val_labels = val_data
    train = TensorDataset(preprocess(train_images, 28, [0.1307], [0.3081]), torch.as_tensor(train_labels).long())
    val = TensorDataset(preprocess(val_images, 28, [0.1307], [0.3081]), torch.as_tensor(val_labels).long())
    loader = DataLoader(train, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed))
    best_accuracy, best_state = -1, None
    for epoch in range(epochs):
        net.train()
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            _, logits = net(images.to(device))
            F.cross_entropy(logits, labels.to(device)).backward()
            optimizer.step()
        net.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in DataLoader(val, batch_size=128):
                _, logits = net(images.to(device))
                correct += (logits.argmax(-1).cpu() == labels).sum().item()
        accuracy = correct / len(val)
        logger.log({"encoder/val_accuracy": accuracy}, epoch + 1)
        if accuracy > best_accuracy:
            best_accuracy, best_state = accuracy, copy.deepcopy(net.state_dict())
    torch.save({"model": best_state, "val_accuracy": best_accuracy, "seed": seed}, path)


class InceptionEncoder:
    def __init__(self, device):
        from torchvision.models import Inception_V3_Weights, inception_v3
        self.device = device
        # Input normalization is performed explicitly below, exactly once.
        self.net = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).eval().to(device)
        self.fc = self.net.fc
        self.net.fc = nn.Identity()

    def preprocess(self, images):
        return preprocess(images, 299, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    @torch.no_grad()
    def __call__(self, x):
        features = self.net(x)
        return features, F.softmax(self.fc(features), -1)


class CLIPEncoder:
    def __init__(self, device, class_prompts=None):
        from transformers import CLIPModel, CLIPTokenizer
        if not class_prompts:
            raise ValueError("Set metrics.clip_prompts for CLIP classifier scores")
        self.device = device
        name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(name).eval().to(device)
        tokenizer = CLIPTokenizer.from_pretrained(name)
        inputs = tokenizer(class_prompts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            self.text_features = F.normalize(self.model.get_text_features(**inputs), dim=-1)

    def preprocess(self, images):
        return preprocess(images, 224, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])

    @torch.no_grad()
    def __call__(self, x):
        features = self.model.get_image_features(pixel_values=x)
        logits = self.model.logit_scale.exp() * F.normalize(features, dim=-1) @ self.text_features.T
        return features, logits.softmax(-1)


ENCODERS = {"pixels": PixelEncoder, "mnist": MNISTEncoder, "inception": InceptionEncoder, "clip": CLIPEncoder}
