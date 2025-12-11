"""Download MNIST dataset into data/raw"""
from torchvision import datasets, transforms
from pathlib import Path

def download():
    out = Path(__file__).resolve().parents[1] / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MNIST into {out}")
    datasets.MNIST(str(out), download=True, train=True, transform=transforms.ToTensor())
    datasets.MNIST(str(out), download=True, train=False, transform=transforms.ToTensor())

if __name__ == '__main__':
    download()
