"""
model.py -- Encoder/Decoder architecture for image watermarking.

Fixes vs. the original prototype:
  * LogoEncoder now outputs the same number of channels as ImageEncoder
    (64 by default) so the two feature maps can actually be summed before
    being handed to the Decoder. In the original code LogoEncoder output
    3 channels while ImageEncoder output 64 -- `F1 + alpha * F2` could not
    broadcast and would crash.
  * ImageEncoder's final BatchNorm2d channel count now matches its Conv2d
    output (64), not the stray `64 + 3`.
"""
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.encoder(x)


class LogoEncoder(nn.Module):
    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """Produces the watermarked RGB image from the fused feature map."""

    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


class LogoExtractor(nn.Module):
    """Recovers the logo from a (possibly attacked) watermarked image."""

    def __init__(self):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.extractor(x)


class WatermarkAutoencoder(nn.Module):
    def __init__(self, alpha: float = 0.1, feature_channels: int = 64):
        super().__init__()
        self.alpha = alpha
        self.image_encoder = ImageEncoder(feature_channels)
        self.logo_encoder = LogoEncoder(feature_channels)
        self.decoder = Decoder(feature_channels)
        self.extractor = LogoExtractor()

    def forward(self, image, logo, noise_layer=None):
        f1 = self.image_encoder(image)
        f2 = self.logo_encoder(logo)
        fused = f1 + self.alpha * f2
        watermarked = self.decoder(fused)

        # Route through a simulated-attack layer during training so the
        # extractor learns to recover the logo even after the watermarked
        # image gets blurred, cropped, compressed, etc.
        attacked = noise_layer(watermarked) if noise_layer is not None else watermarked
        recovered_logo = self.extractor(attacked)
        return watermarked, recovered_logo
