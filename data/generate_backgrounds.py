import os
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
from scipy.signal import butter, filtfilt

PIANO_DIR = "data/piano_clean"
BG_DIR = "data/backgrounds"
SR = 44100


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def highpass(audio, freq=300, sr=SR):
    """Remove most piano low-frequency energy."""
    b, a = butter(4, freq / (sr / 2), btype='high')
    return filtfilt(b, a, audio, axis=1)


def pink_noise(length):
    """Pink noise generator."""
    uneven = length % 2
    X = np.random.randn(length // 2 + 1 + uneven) + 1j * np.random.randn(length // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.)[::-1]
    y = (np.fft.irfft(X * S)).real
    if uneven:
        y = y[:-1]
    y /= np.max(np.abs(y))
    return y


def process_backgrounds():
    ensure_dir(BG_DIR)
    piano_files = [os.path.join(PIANO_DIR, f) for f in os.listdir(PIANO_DIR) if f.endswith(".wav")]

    out_id = 0

    for pf in tqdm(piano_files, desc="Generating backgrounds"):
        y, _ = librosa.load(pf, sr=SR, mono=False)

        if y.ndim == 1:
            y = np.stack([y, y])

        # 1. High-pass filter to remove piano fundamentals
        bg = highpass(y)

        # 2. Add pink noise for realism
        noise = pink_noise(bg.shape[1])
        noise = np.vstack([noise, noise])

        weight = np.random.uniform(0.1, 0.4)
        bg = bg + weight * noise

        # 3. Normalize
        peak = np.max(np.abs(bg))
        if peak > 1e-6:
            bg /= peak

        out = os.path.join(BG_DIR, f"bg_{out_id:05d}.wav")
        sf.write(out, bg.T, SR)
        out_id += 1

    print(f"\nDone! Generated {out_id} background files.")


def main():
    process_backgrounds()


if __name__ == "_main_":
    main()