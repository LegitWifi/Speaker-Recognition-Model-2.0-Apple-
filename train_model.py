import os
import numpy as np
import librosa
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_DIR  = "Dataset"
MODEL_PATH   = "speaker_model.h5"
ENCODER_PATH = "label_encoder.pkl"

SAMPLE_RATE  = 22050
N_MELS       = 128
FIXED_WIDTH  = 128   # fixed time-axis size for the spectrogram
EPOCHS       = 30
BATCH_SIZE   = 4
# ──────────────────────────────────────────────────────────────────────────────


def extract_features(file_path):
    """
    Load a WAV file and return:
      - mel_spec  : Mel spectrogram resized to (N_MELS, FIXED_WIDTH, 1)
      - pitch_feats: [mean, std, min, max] of the pitch (F0) curve
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # ---------- Mel Spectrogram ----------
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)          # convert to dB scale

    # Pad or trim to fixed width
    if mel_db.shape[1] < FIXED_WIDTH:
        pad_width = FIXED_WIDTH - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :FIXED_WIDTH]

    mel_db = mel_db[:, :, np.newaxis]                       # shape: (128, 128, 1)

    # ---------- Pitch (F0) ----------
    f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    f0 = f0[f0 > 0]                                         # keep only voiced frames

    if len(f0) == 0:
        pitch_feats = np.array([0.0, 0.0, 0.0, 0.0])
    else:
        pitch_feats = np.array([
            float(np.mean(f0)),
            float(np.std(f0)),
            float(np.min(f0)),
            float(np.max(f0))
        ])

    return mel_db, pitch_feats


def load_dataset():
    """Walk through Dataset/ and collect features + labels."""
    X_mel, X_pitch, labels = [], [], []

    speakers = sorted(os.listdir(DATASET_DIR))
    print(f"\nFound speakers: {speakers}\n")

    for speaker in speakers:
        speaker_path = os.path.join(DATASET_DIR, speaker)
        if not os.path.isdir(speaker_path):
            continue

        wav_files = [f for f in os.listdir(speaker_path) if f.endswith(".wav")]
        print(f"  [{speaker}]  →  {len(wav_files)} files")

        for wav_file in wav_files:
            file_path = os.path.join(speaker_path, wav_file)
            mel, pitch = extract_features(file_path)
            X_mel.append(mel)
            X_pitch.append(pitch)
            labels.append(speaker)

    return np.array(X_mel), np.array(X_pitch), np.array(labels)


def build_cnn(num_classes):
    """Simple beginner-friendly CNN."""
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(N_MELS, FIXED_WIDTH, 1)),
        MaxPooling2D((2, 2)),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        # Classifier head
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    print("=" * 50)
    print("   SPEAKER RECOGNITION — TRAINING")
    print("=" * 50)

    # 1. Load data
    print("\n[1] Loading dataset and extracting features...")
    X_mel, X_pitch, labels = load_dataset()
    print(f"\nTotal samples: {len(labels)}")

    # 2. Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)
    print(f"Classes: {list(le.classes_)}")

    y_cat = to_categorical(y_encoded, num_classes=num_classes)

    # 3. Train/Test split
    X_mel_train, X_mel_test, y_train, y_test = train_test_split(
        X_mel, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"\nTrain samples: {len(X_mel_train)}  |  Test samples: {len(X_mel_test)}")

    # 4. Build & train model
    print("\n[2] Building CNN...")
    model = build_cnn(num_classes)
    model.summary()

    print("\n[3] Training...\n")
    model.fit(
        X_mel_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_mel_test, y_test),
        verbose=1
    )

    # 5. Evaluate
    loss, acc = model.evaluate(X_mel_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {acc * 100:.1f}%")

    # 6. Save
    model.save(MODEL_PATH)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    print(f"\n[✓] Model saved  →  {MODEL_PATH}")
    print(f"[✓] Encoder saved →  {ENCODER_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()
