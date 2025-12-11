"""Train a small CNN on the processed MNIST dataset and save a checkpoint to models/"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


def load_data(processed_path, batch_size):
    train_images, train_labels = torch.load(processed_path / 'train.pt')
    dataset = TensorDataset(train_images, train_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def train(args):
    processed = Path(__file__).resolve().parents[1] / 'data' / 'processed'
    models_dir = Path(__file__).resolve().parents[1] / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    train_loader = load_data(processed, args.batch_size)

    model = SmallCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        epoch_loss = running / len(train_loader.dataset)
        print(f"Epoch {epoch}/{args.epochs} - loss: {epoch_loss:.4f} - time: {time.time()-start:.1f}s")

    ckpt = models_dir / 'model.pth'
    torch.save(model.state_dict(), ckpt)
    print(f"Saved model checkpoint to {ckpt}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
