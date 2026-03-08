"""
generate_spectrograms.py
------------------------
Generates a Mel-spectrogram PNG image for every .wav file found under
DATASET_DIR and saves them in a mirrored folder structure inside OUTPUT_DIR.

Output layout:
    Spectrograms/
        Rishav_Apple/
            audio1.png
            audio2.png
            ...
        Utkrarsh_apple/
            audio1.png
            ...
        sawant_apple/
            audio1.png
            ...

Usage:
    python generate_spectrograms.py
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – no GUI window needed
import matplotlib.pyplot as plt

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_DIR  = "Dataset"       # folder that contains one sub-folder per speaker
OUTPUT_DIR   = "Spectrograms"  # root folder where images will be saved

SAMPLE_RATE  = 22050           # Hz  (must match train_model.py)
N_MELS       = 128             # number of Mel filter banks
HOP_LENGTH   = 512             # samples between successive frames
FIG_SIZE     = (6, 4)          # inches  (width, height) of each saved image
DPI          = 100             # dots-per-inch of the saved PNG
# ───────────────────────────────────────────────────────────────────────────────


def generate_spectrogram(wav_path: str, out_path: str) -> None:
    """
    Load a WAV file, compute its Mel spectrogram (in dB) and save it as a PNG.

    Parameters
    ----------
    wav_path : str  – absolute or relative path to the source .wav file
    out_path : str  – path where the output .png will be written
    """
    # 1. Load audio
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

    # 2. Compute Mel spectrogram and convert to dB scale
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 3. Plot
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(os.path.splitext(os.path.basename(wav_path))[0])
    plt.tight_layout()

    # 4. Save and close (freeing memory)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def main() -> None:
    print("=" * 55)
    print("   SPECTROGRAM GENERATOR")
    print("=" * 55)

    if not os.path.isdir(DATASET_DIR):
        print(f"[ERROR] Dataset directory not found: '{DATASET_DIR}'")
        print("        Make sure you run this script from the project root.")
        return

    total_saved   = 0
    total_skipped = 0

    # Walk through each speaker sub-folder
    speakers = sorted(os.listdir(DATASET_DIR))
    print(f"\nFound {len(speakers)} speaker folder(s): {speakers}\n")

    for speaker in speakers:
        speaker_src = os.path.join(DATASET_DIR, speaker)
        if not os.path.isdir(speaker_src):
            continue                                  # skip stray files

        # Mirror folder inside OUTPUT_DIR
        speaker_out = os.path.join(OUTPUT_DIR, speaker)
        os.makedirs(speaker_out, exist_ok=True)

        wav_files = [f for f in os.listdir(speaker_src) if f.lower().endswith(".wav")]
        print(f"  [{speaker}]  →  {len(wav_files)} WAV file(s) found")

        for wav_file in wav_files:
            wav_path = os.path.join(speaker_src, wav_file)
            png_name = os.path.splitext(wav_file)[0] + ".png"
            out_path = os.path.join(speaker_out, png_name)

            if os.path.exists(out_path):
                print(f"      [SKIP]  {png_name}  (already exists)")
                total_skipped += 1
                continue

            try:
                generate_spectrogram(wav_path, out_path)
                print(f"      [OK]    {png_name}")
                total_saved += 1
            except Exception as exc:
                print(f"      [FAIL]  {wav_file}  →  {exc}")

    print()
    print(f"Done!  Saved: {total_saved}  |  Skipped: {total_skipped}")
    print(f"Spectrograms stored in: '{os.path.abspath(OUTPUT_DIR)}'")
    print("=" * 55)


if __name__ == "__main__":
    main()
