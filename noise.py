"""
noise.py -- Simulated-attack layer.

Training the extractor with these perturbations in the loop is what makes
this a *robust* watermark rather than one that only survives if nobody
touches the image. This is the standard trick used in real deep-learning
watermarking papers (e.g. HiDDeN).
"""
import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class NoiseLayer(torch.nn.Module):
    def __init__(self, p_identity: float = 0.2):
        super().__init__()
        self.p_identity = p_identity

    def forward(self, x):
        if not self.training:
            return x
        r = random.random()
        if r < self.p_identity:
            return x
        elif r < 0.4:
            return self._gaussian_noise(x)
        elif r < 0.6:
            return self._blur(x)
        elif r < 0.8:
            return self._dropout(x)
        else:
            return self._crop_resize(x)

    @staticmethod
    def _gaussian_noise(x, std: float = 0.05):
        return torch.clamp(x + torch.randn_like(x) * std, 0, 1)

    @staticmethod
    def _blur(x):
        return TF.gaussian_blur(x, kernel_size=3)

    @staticmethod
    def _dropout(x, p: float = 0.1):
        mask = (torch.rand_like(x[:, :1]) > p).float()
        return x * mask

    @staticmethod
    def _crop_resize(x, scale: float = 0.8):
        n, c, h, w = x.shape
        nh, nw = int(h * scale), int(w * scale)
        top = random.randint(0, h - nh)
        left = random.randint(0, w - nw)
        cropped = x[:, :, top:top + nh, left:left + nw]
        return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)
