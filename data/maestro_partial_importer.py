
import os
import tarfile
import urllib.request
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.tar.gz"
DOWNLOAD_PATH = "maestro-v3.0.0.tar.gz"
EXTRACT_DIR = "maestro"
OUTPUT_DIR = "data/piano_clean"
TARGET_SR = 44100
MAX_FILES = 150           # ~5GB of audio


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def download_maestro():
    if os.path.exists(DOWNLOAD_PATH):
        print("Archive already downloaded.")
        return

    print("Downloading MAESTRO v3 (34GB archive)…")
    urllib.request.urlretrieve(
        MAESTRO_URL,
        DOWNLOAD_PATH,
        reporthook=lambda b, bs, t: print(f"\r{b * bs / t * 100:.2f}%   ", end="")
    )
    print("\nDownload complete!")


def extract_few_wavs(max_files=MAX_FILES):
    print(f"Extracting only {max_files} WAV files (~5GB)…")

    ensure_dir(EXTRACT_DIR)

    count = 0
    with tarfile.open(DOWNLOAD_PATH, "r:gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".wav"):
                tar.extract(member, EXTRACT_DIR)
                count += 1

                if count >= max_files:
                    print(f"Extracted {count} WAV files.")
                    return

    print(f"Done. Extracted {count} files.")


def load_audio_stereo(path, sr=TARGET_SR):
    audio, _ = librosa.load(path, sr=sr, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    return audio


def normalize(a):
    peak = np.max(np.abs(a))
    return a if peak < 1e-6 else a / peak


def convert_to_clean_piano():
    ensure_dir(OUTPUT_DIR)

    wavs = []
    for root, _, files in os.walk(EXTRACT_DIR):
        for f in files:
            if f.endswith(".wav"):
                wavs.append(os.path.join(root, f))

    print(f"Processing {len(wavs)} WAV files as piano_clean…")

    for i, path in enumerate(tqdm(wavs)):
        try:
            audio = load_audio_stereo(path)
            audio = normalize(audio)
            out = os.path.join(OUTPUT_DIR, f"piano_{i:05d}.wav")
            sf.write(out, audio.T, TARGET_SR)
        except Exception as e:
            print(f"Error processing {path}: {e}")


def main():
    ensure_dir("data")
    download_maestro()
    extract_few_wavs()
    convert_to_clean_piano()
    print("\nPartial MAESTRO piano dataset ready!")


if __name__ == "__main__":
    main()