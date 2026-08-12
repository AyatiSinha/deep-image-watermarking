# Deep Image Watermarking

A deep learning-based image watermarking system that embeds and recovers a logo from images using a CNN-based encoder-decoder architecture. The project evaluates the trade-off between image fidelity and watermark recovery and investigates robustness under simulated noise-based attacks.

## Overview

Digital images can be copied, modified, and redistributed easily, creating a need for techniques that can embed ownership information while preserving image quality.

This project implements a neural image watermarking pipeline that learns to:

- Embed a logo into an image
- Generate a visually similar watermarked image
- Recover the embedded logo
- Measure image fidelity
- Measure watermark recovery accuracy
- Evaluate robustness under simulated noise

The system is implemented using PyTorch and convolutional neural networks.

## Methodology

The watermarking pipeline consists of four main components:

```text
                 Original Image
                       │
                       ▼
                Image Encoder
                       │
                       │
Logo ─────────► Logo Encoder
                       │
                       ▼
                Feature Fusion
                       │
                       ▼
                   Decoder
                       │
                       ▼
               Watermarked Image
                       │
                       ▼
                Logo Extractor
                       │
                       ▼
                Recovered Logo
