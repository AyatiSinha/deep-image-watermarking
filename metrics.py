import torch


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """Peak signal-to-noise ratio between watermarked and original images, in dB.
    Higher is better; >30dB is generally considered visually near-identical.
    """
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    return 10 * torch.log10((max_val ** 2) / (mse + eps))


def bit_accuracy(pred_logo: torch.Tensor, true_logo: torch.Tensor, threshold: float = 0.5) -> float:
    """Fraction of pixels where the extracted logo agrees with the ground
    truth after binarizing at `threshold`."""
    pred_bits = (pred_logo > threshold).float()
    true_bits = (true_logo > threshold).float()
    return (pred_bits == true_bits).float().mean().item()
