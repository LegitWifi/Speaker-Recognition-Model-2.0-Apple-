import os
import pickle
import threading
import numpy as np
import librosa
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
from tensorflow.keras.models import load_model

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH   = "speaker_model.h5"
ENCODER_PATH = "label_encoder.pkl"
SAMPLE_RATE  = 22050
N_MELS       = 128
FIXED_WIDTH  = 128
TEMP_WAV     = "temp_recording.wav"
# ──────────────────────────────────────────────────────────────────────────────


def extract_features(file_path):
    """
    Extract the same features used during training:
      - Mel spectrogram resized to (N_MELS, FIXED_WIDTH, 1)
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Pad or trim to fixed width
    if mel_db.shape[1] < FIXED_WIDTH:
        pad_width = FIXED_WIDTH - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :FIXED_WIDTH]

    mel_db = mel_db[:, :, np.newaxis]                     # (128, 128, 1)
    return mel_db


class Recorder:
    """Handles non-blocking microphone recording."""

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.recording = False
        self.frames = []

    def _callback(self, indata, frames, time, status):
        if self.recording:
            self.frames.append(indata.copy())

    def start(self):
        self.frames = []
        self.recording = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self._callback
        )
        self.stream.start()

    def stop(self):
        self.recording = False
        self.stream.stop()
        self.stream.close()
        if len(self.frames) == 0:
            return None
        audio = np.concatenate(self.frames, axis=0)
        return audio


def save_wav(audio, path, sample_rate):
    audio_int = (audio * 32767).astype(np.int16)
    wav_write(path, sample_rate, audio_int)


def predict_speaker(model, label_encoder, file_path):
    """Run inference and return predicted speaker name."""
    mel = extract_features(file_path)
    mel_input = np.expand_dims(mel, axis=0)               # (1, 128, 128, 1)
    predictions = model.predict(mel_input, verbose=0)
    predicted_class = np.argmax(predictions)
    speaker_name = label_encoder.inverse_transform([predicted_class])[0]
    confidence = predictions[0][predicted_class] * 100
    return speaker_name, confidence


def main():
    print("=" * 50)
    print("   SPEAKER RECOGNITION — LIVE DEMO")
    print("=" * 50)

    # Load model and encoder
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model not found: '{MODEL_PATH}'")
        print("  Please run  train_model.py  first.")
        return

    print("\n[✓] Loading model...")
    model = load_model(MODEL_PATH)

    print("[✓] Loading label encoder...")
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    print(f"\nKnown speakers: {list(label_encoder.classes_)}")
    print("=" * 50)

    recorder = Recorder(SAMPLE_RATE)

    while True:
        user_input = input("\nType 'start' to record  |  'stop' to exit: ").strip().lower()

        if user_input == "stop":
            print("Goodbye!")
            break

        if user_input != "start":
            print("  → Please type 'start' to begin.")
            continue

        # ── Start Recording ──────────────────────────────────────────────────
        print("\n🎙  Recording...  (type 'stop' and press Enter to finish)")
        recorder.start()

        # Wait for user to type "stop" in a separate input call
        stop_input = input("").strip().lower()
        while stop_input != "stop":
            stop_input = input("  → Type 'stop' to stop recording: ").strip().lower()

        audio = recorder.stop()
        # ────────────────────────────────────────────────────────────────────

        if audio is None or len(audio) == 0:
            print("[!] No audio captured. Please try again.")
            continue

        duration = len(audio) / SAMPLE_RATE
        print(f"\n[✓] Captured {duration:.1f} seconds of audio.")

        # Save to temp file and predict
        save_wav(audio, TEMP_WAV, SAMPLE_RATE)
        print("[✓] Processing audio...")

        speaker, confidence = predict_speaker(model, label_encoder, TEMP_WAV)

        print("\n" + "=" * 50)
        print(f"  ✅  Matched Speaker : {speaker}")
        print(f"  📊  Confidence      : {confidence:.1f}%")
        print("=" * 50)

        # Clean up temp file
        if os.path.exists(TEMP_WAV):
            os.remove(TEMP_WAV)


if __name__ == "__main__":
    main()
