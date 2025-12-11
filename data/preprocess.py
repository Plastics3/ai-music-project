"""Preprocess MNIST and save processed tensors to data/processed

Produces `data/processed/train.pt` and `data/processed/test.pt` as tuples (images, labels).
Images are float32 tensors normalized with MNIST mean/std.
"""
import torch
from torchvision import datasets, transforms
from pathlib import Path

def preprocess():
    base = Path(__file__).resolve().parents[1] / "data"
    raw = base / "raw"
    out = base / "processed"
    out.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("Loading raw MNIST datasets (this will not re-download if already present)")
    train_ds = datasets.MNIST(str(raw), train=True, download=False, transform=transform)
    test_ds = datasets.MNIST(str(raw), train=False, download=False, transform=transform)

    print("Converting to tensors and saving")
    train_images = torch.stack([img for img, _ in train_ds]).to(torch.float32)
    train_labels = torch.tensor([label for _, label in train_ds], dtype=torch.long)
    test_images = torch.stack([img for img, _ in test_ds]).to(torch.float32)
    test_labels = torch.tensor([label for _, label in test_ds], dtype=torch.long)

    torch.save((train_images, train_labels), out / 'train.pt')
    torch.save((test_images, test_labels), out / 'test.pt')
    print(f"Saved processed datasets to {out}")

if __name__ == '__main__':
    preprocess()
