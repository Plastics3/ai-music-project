import torch
import onnx
import argparse
from demucs.pretrained import load_model

def export(model_path, output_path):
    print("Loading model:", model_path)
    model = load_model(model_path)
    model.eval()

    # Dummy input: (batch=1, channels=2, samples=44100*6)
    dummy = torch.randn(1, 2, 44100 * 6, dtype=torch.float32)

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["waveform"],
        output_names=["piano"],
        dynamic_axes={
            "waveform": {2: "samples"},
            "piano": {2: "samples"}
        },
        opset_version=16
    )

    print("Done! Saved to:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to Demucs .th checkpoint")
    parser.add_argument("--out", default="piano.onnx",
                        help="Output ONNX file")
    args = parser.parse_args()

    export(args.model, args.out)