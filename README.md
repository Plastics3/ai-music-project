AI Project
===========

This repository contains a small example AI project with four sections:

1) Data: download and preprocess the dataset
2) Train: training script that saves a PyTorch checkpoint
3) Export: export the trained model to ONNX
4) Application: simple Flask app that serves an ONNX model for inference

Quick steps (PowerShell):

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download dataset:

```powershell
python data\download_dataset.py
python data\preprocess.py
```

Train (short example):

```powershell
python train\train.py --epochs 1 --batch-size 64
```

Export to ONNX:

```powershell
python export\export_onnx.py --checkpoint models\model.pth --output models\model.onnx
```

Run the app (default port 5000):

```powershell
python app\app.py
```

Then POST an image to `http://localhost:5000/predict` as `multipart/form-data` with key `image`.

Notes:
- This example uses MNIST and a tiny CNN. Adjust scripts for your dataset and model.
- See individual scripts for options and implementation details.
