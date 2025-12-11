"""Simple Flask app that loads ONNX model and serves predictions.

POST /predict with multipart-form `image` to get a predicted label (MNIST 28x28 grayscale image).
"""
from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import onnxruntime as ort
from pathlib import Path

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'model.onnx'

session = None

def load_model(path):
    print(f"Loading ONNX model from {path}")
    return ort.InferenceSession(str(path))

def preprocess_image(file_stream):
    img = Image.open(file_stream).convert('L').resize((28,28))
    arr = np.array(img).astype(np.float32) / 255.0
    # normalize with MNIST mean/std used in preprocess
    arr = (arr - 0.1307) / 0.3081
    arr = arr[np.newaxis, np.newaxis, :, :]
    return arr

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error':'no image file provided, use key `image` in multipart/form-data'}), 400
    f = request.files['image']
    arr = preprocess_image(f.stream)
    inputs = {session.get_inputs()[0].name: arr}
    out = session.run(None, inputs)
    logits = out[0]
    pred = int(np.argmax(logits, axis=1)[0])
    return jsonify({'pred': pred})

if __name__ == '__main__':
    if not MODEL_PATH.exists():
        print(f"ONNX model not found at {MODEL_PATH}. Run export/export_onnx.py first.")
        exit(1)
    session = load_model(MODEL_PATH)
    app.run(host='0.0.0.0', port=5000)
