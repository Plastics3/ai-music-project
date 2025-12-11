import os
import random
import librosa
import soundfile as sf
import numpy as np

PIANO_DIR = "data/piano_clean"            # folder containing isolated piano stems
BG_DIR = "data/backgrounds"               # folder with unrelated music/noise/instruments
OUTPUT_DIR = "data/train"                 # output dataset
SR = 44100                                # sampling rate for training
MIN_DUR = 5                               # minimum duration (seconds)
SNR_RANGE = (-5, 5)                       # random SNR (dB) for piano vs bg


def load_audio(path, sr=SR):
    audio, _ = librosa.load(path, sr=sr, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)  # mono → stereo
    return audio


def normalize(audio):
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return audio
    return audio / peak


def mix_at_snr(target, bg, snr_db):
    # match lengths
    min_len = min(target.shape[1], bg.shape[1])
    target = target[:, :min_len]
    bg = bg[:, :min_len]

    # convert SNR(dB) → linear gain
    snr = 10 ** (snr_db / 20)

    # adjust background RMS
    rms_t = np.sqrt(np.mean(target**2))
    rms_b = np.sqrt(np.mean(bg**2))

    if rms_b > 0:
        bg = bg * (rms_t / (rms_b * snr))

    mixture = target + bg
    return mixture, target


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    ensure_dir(OUTPUT_DIR)

    piano_files = [f for f in os.listdir(PIANO_DIR) if f.endswith(".wav")]
    bg_files = [f for f in os.listdir(BG_DIR) if f.endswith(".wav")]

    if not piano_files:
        print("No piano files found!")
        return
    if not bg_files:
        print("No background files found!")
        return

    count = 0
    for pf in piano_files:
        piano_path = os.path.join(PIANO_DIR, pf)

        # load piano
        piano = load_audio(piano_path)
        piano = normalize(piano)

        # skip very short samples
        if piano.shape[1] < MIN_DUR * SR:
            continue

        # pick a random background
        bg_path = os.path.join(BG_DIR, random.choice(bg_files))
        bg = load_audio(bg_path)
        bg = normalize(bg)

        # random SNR
        snr_db = random.uniform(*SNR_RANGE)

        mix, tgt = mix_at_snr(piano, bg, snr_db)
        mix = normalize(mix)

        # output folder
        out_dir = os.path.join(OUTPUT_DIR, f"sample_{count}")
        ensure_dir(out_dir)

        # write files
        sf.write(os.path.join(out_dir, "mixture.wav"), mix.T, SR)
        sf.write(os.path.join(out_dir, "piano.wav"), tgt.T, SR)

        print(f"[OK] Created sample {count} (SNR={snr_db:.2f} dB)")
        count += 1

    print(f"\nDone! Created {count} training pairs.")


if __name__ == "__main__":
    main()