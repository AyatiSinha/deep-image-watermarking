"""
dataset.py -- Pairs CIFAR-10 images with a watermark logo.
"""
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

LOGO_TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])


class WatermarkDataset(Dataset):
    def __init__(self, image_dataset, logo_tensor):
        self.dataset = image_dataset
        self.logo = logo_tensor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image, self.logo


def build_datasets(data_root: str, logo_path: str):
    """Returns (train_dataset, val_dataset, logo_tensor).

    Uses CIFAR-10's real train/test split -- the original script trained
    and "evaluated" on the exact same images, which tells you nothing
    about whether the model generalizes.
    """
    train_base = datasets.CIFAR10(root=data_root, train=True, download=False, transform=IMAGE_TRANSFORM)
    val_base = datasets.CIFAR10(root=data_root, train=False, download=False, transform=IMAGE_TRANSFORM)

    logo_img = Image.open(logo_path).convert("RGB")
    logo_tensor = LOGO_TRANSFORM(logo_img)

    train_ds = WatermarkDataset(train_base, logo_tensor)
    val_ds = WatermarkDataset(val_base, logo_tensor)
    return train_ds, val_ds, logo_tensor
