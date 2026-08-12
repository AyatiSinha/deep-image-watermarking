import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import build_datasets
from metrics import bit_accuracy, psnr
from model import WatermarkAutoencoder
from noise import NoiseLayer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data")
    p.add_argument("--logo", default="logo.png")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.1, help="weight of logo features in the fused map")
    p.add_argument("--beta", type=float, default=1.0, help="image reconstruction loss weight")
    p.add_argument("--gamma", type=float, default=10.0, help="logo recovery loss weight")
    p.add_argument("--use-noise-layer", action="store_true", help="train against simulated attacks")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="./outputs")
    return p.parse_args()


def evaluate(model, loader, device, noise_layer):
    model.eval()
    total_psnr, total_bit_acc, n_batches = 0.0, 0.0, 0
    with torch.no_grad():
        for image, logo in loader:
            image, logo = image.to(device), logo.to(device)
            watermarked, recovered = model(image, logo, noise_layer)
            total_psnr += psnr(watermarked, image).mean().item()
            total_bit_acc += bit_accuracy(recovered, logo)
            n_batches += 1
    model.train()
    return total_psnr / n_batches, total_bit_acc / n_batches


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, _ = build_datasets(args.data_root, args.logo)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = WatermarkAutoencoder(alpha=args.alpha).to(device)
    noise_layer = NoiseLayer() if args.use_noise_layer else None

    image_loss_fn = nn.MSELoss()
    logo_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_psnr": [], "val_bit_acc": []}
    best_val_psnr = -1.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for image, logo in train_loader:
            image, logo = image.to(device), logo.to(device)
            watermarked, recovered = model(image, logo, noise_layer)

            l1 = image_loss_fn(watermarked, image)
            l2 = logo_loss_fn(recovered, logo)
            loss = args.beta * l1 + args.gamma * l2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        val_psnr, val_bit_acc = evaluate(model, val_loader, device, noise_layer)

        history["train_loss"].append(avg_train_loss)
        history["val_psnr"].append(val_psnr)
        history["val_bit_acc"].append(val_bit_acc)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_PSNR={val_psnr:.2f}dB  val_bit_acc={val_bit_acc:.4f}"
        )

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))

    torch.save(model.state_dict(), os.path.join(args.out_dir, "final_model.pt"))
    save_history_plot(history, args.out_dir)
    save_sample_grid(model, val_loader, device, noise_layer, args.out_dir)
    print(f"\nDone. Checkpoints and plots saved to {args.out_dir}/")


def save_history_plot(history, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"])
    axes[0].set_title("Train loss")
    axes[1].plot(history["val_psnr"])
    axes[1].set_title("Val PSNR (dB)")
    axes[2].plot(history["val_bit_acc"])
    axes[2].set_title("Val logo bit accuracy")
    for ax in axes:
        ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def save_sample_grid(model, loader, device, noise_layer, out_dir, n=5):
    model.eval()
    image, logo = next(iter(loader))
    image, logo = image[:n].to(device), logo[:n].to(device)
    with torch.no_grad():
        watermarked, recovered = model(image, logo, noise_layer)

    fig, axes = plt.subplots(n, 4, figsize=(10, 2.5 * n))
    for i in range(n):
        axes[i, 0].imshow(image[i].cpu().permute(1, 2, 0))
        axes[i, 0].set_title("Original")
        axes[i, 1].imshow(watermarked[i].cpu().permute(1, 2, 0))
        axes[i, 1].set_title("Watermarked")
        axes[i, 2].imshow(logo[i].cpu().squeeze(), cmap="gray")
        axes[i, 2].set_title("True logo")
        axes[i, 3].imshow(recovered[i].cpu().squeeze(), cmap="gray")
        axes[i, 3].set_title("Recovered")
        for j in range(4):
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sample_predictions.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
