"""Load checkpoint and export the model to ONNX format."""
import argparse
from pathlib import Path
import torch
import sys

# Import the same model definition as in train
sys.path.append(str(Path(__file__).resolve().parents[1] / 'train'))
from train import SmallCNN  # type: ignore


def export(checkpoint, output):
    device = torch.device('cpu')
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    dummy = torch.randn(1, 1, 28, 28, dtype=torch.float32)
    torch.onnx.export(model, dummy, str(output), export_params=True, opset_version=11,
                      input_names=['input'], output_names=['output'])
    print(f"Exported ONNX model to {output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=str(Path(__file__).resolve().parents[1] / 'models' / 'model.pth'))
    parser.add_argument('--output', type=str, default=str(Path(__file__).resolve().parents[1] / 'models' / 'model.onnx'))
    args = parser.parse_args()
    export(args.checkpoint, args.output)
