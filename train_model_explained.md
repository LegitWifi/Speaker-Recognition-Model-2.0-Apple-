# 📘 train_model.py — Beginner's Line-by-Line Guide

This document explains **every single line** of `train_model.py` in plain English.
No prior machine-learning knowledge is needed!

---

## 🗺️ Big Picture (What the whole file does)

```
Your WAV files  →  Extract Mel Spectrograms  →  Feed into CNN  →  Trained Model saved to disk
```

The script teaches a **Convolutional Neural Network (CNN)** to recognise different speakers
by looking at the "sound pictures" (spectrograms) of their voice recordings.

---

## 📦 Section 1 — Imports (Lines 1–9)

```python
import os
```
> `os` is a built-in Python module that lets us work with files and folders
> (e.g. listing files in a directory).

```python
import numpy as np
```
> `numpy` is the go-to library for fast number crunching.
> We use it to handle arrays of audio data and features.
> `as np` is just a shortcut so we type `np` instead of `numpy` every time.

```python
import librosa
```
> `librosa` is a specialist audio-analysis library.
> It can load WAV files and compute spectrograms for us.

```python
import pickle
```
> `pickle` saves Python objects to a file and loads them back later.
> We use it to save the **label encoder** (the mapping of speaker names ↔ numbers).

```python
from sklearn.preprocessing import LabelEncoder
```
> Computers work with numbers, not text.
> `LabelEncoder` converts speaker names like `"Rishav_Apple"` into integers like `0, 1, 2`.

```python
from sklearn.model_selection import train_test_split
```
> Splits our dataset into a **training set** (model learns from this)
> and a **test set** (we check accuracy on this — the model has never seen it).

```python
from tensorflow.keras.models import Sequential
```
> `Sequential` is a type of neural-network model where layers are stacked one after another,
> like floors of a building.

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
```
> These are the individual **building blocks** (layers) of the CNN:
> - `Conv2D` — detects patterns in a 2-D image (our spectrogram)
> - `MaxPooling2D` — shrinks the image to keep only the strongest features
> - `Flatten` — converts the 2-D feature map into a 1-D list of numbers
> - `Dense` — a traditional fully-connected neural-network layer
> - `Dropout` — randomly switches off some neurons during training to prevent overfitting

```python
from tensorflow.keras.utils import to_categorical
```
> Converts integer class labels (e.g. `0, 1, 2`) into **one-hot vectors**
> (e.g. `[1,0,0]`, `[0,1,0]`, `[0,0,1]`) — the format Keras needs.

---

## ⚙️ Section 2 — Configuration Constants (Lines 11–21)

```python
DATASET_DIR  = "Dataset"
```
> The folder name that contains our speaker sub-folders.
> Using a constant at the top means we only need to change it in one place.

```python
MODEL_PATH   = "speaker_model.h5"
```
> The filename where the trained model will be saved.
> `.h5` is a standard format for Keras/TensorFlow models.

```python
ENCODER_PATH = "label_encoder.pkl"
```
> The filename where the label encoder will be saved.
> `.pkl` is the standard extension for pickle files.

```python
SAMPLE_RATE  = 22050
```
> Audio is sampled 22,050 times per second (standard for music/speech analysis).
> Every WAV file is loaded at this rate for consistency.

```python
N_MELS       = 128
```
> The spectrogram will have **128 Mel frequency bands** (rows in the image).
> More bands = more frequency detail, but more computation.

```python
FIXED_WIDTH  = 128
```
> The spectrogram will be forced to exactly **128 time steps wide** (columns).
> Neural networks need all inputs to be the same shape.

```python
EPOCHS       = 30
```
> The model will see the entire training dataset **30 times** during training.
> More epochs generally means better learning (up to a point).

```python
BATCH_SIZE   = 4
```
> Instead of updating the model after every single sample, it processes
> **4 samples at a time** before updating weights.
> Small batches work well when we have very few samples (15 total here).

---

## 🔬 Section 3 — `extract_features()` Function (Lines 24–59)

This is the **most important function**. It turns a raw WAV file into numbers the CNN can learn from.

```python
def extract_features(file_path):
```
> Defines a function that takes the path to a WAV file as input.

```python
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
```
> Loads the audio file.
> - `y` = a NumPy array of audio amplitude values (the actual sound wave)
> - `sr` = the sample rate (will be 22,050 because we forced it)

### 🎨 Mel Spectrogram

```python
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
```
> Converts the raw audio waveform into a **Mel spectrogram**.
>
> 💡 **What is a Mel spectrogram?**
> Think of it like a photo of sound.
> - The **X-axis** is time (left = start of audio, right = end)
> - The **Y-axis** is frequency (bottom = low pitch, top = high pitch)
> - The **colour/brightness** of each pixel shows how loud that frequency is at that moment
>
> "Mel" means the frequencies are spaced the way human ears perceive them
> (we are more sensitive to low frequencies than high ones).

```python
    mel_db = librosa.power_to_db(mel, ref=np.max)
```
> Converts the spectrogram values from raw **power** units to **decibels (dB)**.
> dB scale is logarithmic — it matches how humans actually perceive loudness.
> `ref=np.max` means the loudest point in the file becomes 0 dB and everything else is negative.

### ✂️ Padding / Trimming to Fixed Size

```python
    if mel_db.shape[1] < FIXED_WIDTH:
        pad_width = FIXED_WIDTH - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :FIXED_WIDTH]
```
> The width (number of time frames) of a spectrogram depends on how long the audio clip is.
> Since the CNN expects a fixed size, we either:
> - **Pad** (add zeros to the right) if the clip is too short, or
> - **Trim** (cut off the end) if the clip is too long
>
> After this block, `mel_db` is always exactly (128 rows × 128 columns).

```python
    mel_db = mel_db[:, :, np.newaxis]   # shape becomes (128, 128, 1)
```
> Adds an extra dimension at the end.
> `Conv2D` in Keras expects inputs shaped as `(height, width, channels)`.
> Our spectrogram has only 1 channel (like a grayscale image), so we add a `1`.

### 🎵 Pitch (F0) Features

```python
    f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
```
> Extracts the **fundamental frequency (F0)** — the pitch of the voice over time.
> `fmin=50, fmax=500` limits the range to typical human voice frequencies.
> Result is an array with one pitch value per audio frame.

```python
    f0 = f0[f0 > 0]
```
> Keeps only the **voiced frames** (where actual speech is detected).
> Unvoiced frames (silence or noise) have F0 = 0, so we discard them.

```python
    if len(f0) == 0:
        pitch_feats = np.array([0.0, 0.0, 0.0, 0.0])
    else:
        pitch_feats = np.array([
            float(np.mean(f0)),
            float(np.std(f0)),
            float(np.min(f0)),
            float(np.max(f0))
        ])
```
> Summarises the pitch curve into **4 numbers**: mean, standard deviation, min, max.
> If there are no voiced frames at all, we default to four zeros.

```python
    return mel_db, pitch_feats
```
> Returns both outputs — the spectrogram image and the 4 pitch numbers.

---

## 📂 Section 4 — `load_dataset()` Function (Lines 62–84)

```python
def load_dataset():
    X_mel, X_pitch, labels = [], [], []
```
> Creates three empty lists to collect features and labels from all audio files.

```python
    speakers = sorted(os.listdir(DATASET_DIR))
    print(f"\nFound speakers: {speakers}\n")
```
> Lists all items inside the `Dataset/` folder (each item is a speaker sub-folder).
> `sorted()` puts them in alphabetical order for consistency.

```python
    for speaker in speakers:
        speaker_path = os.path.join(DATASET_DIR, speaker)
        if not os.path.isdir(speaker_path):
            continue
```
> Loops through each speaker folder.
> `os.path.join` builds the full path like `"Dataset/Rishav_Apple"`.
> The `if not os.path.isdir` check skips any stray files that aren't folders.

```python
        wav_files = [f for f in os.listdir(speaker_path) if f.endswith(".wav")]
```
> Lists only the `.wav` files inside each speaker's folder.
> This is called a **list comprehension** — a compact Python way of creating a filtered list.

```python
        for wav_file in wav_files:
            file_path = os.path.join(speaker_path, wav_file)
            mel, pitch = extract_features(file_path)
            X_mel.append(mel)
            X_pitch.append(pitch)
            labels.append(speaker)
```
> For each WAV file:
> 1. Build the full file path
> 2. Extract the Mel spectrogram and pitch features
> 3. Add them to their respective lists
> 4. Record which speaker this belongs to

```python
    return np.array(X_mel), np.array(X_pitch), np.array(labels)
```
> Converts the Python lists into NumPy arrays (needed by Keras)
> and returns all three together.

---

## 🏗️ Section 5 — `build_cnn()` Function (Lines 87–108)

```python
def build_cnn(num_classes):
```
> Builds and returns the CNN model.
> `num_classes` = how many speakers (3 in our case).

```python
    model = Sequential([
```
> Creates a model where layers are stacked in sequence.

### Block 1

```python
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(N_MELS, FIXED_WIDTH, 1)),
```
> **First convolutional layer.**
> - `32` = learns 32 different patterns (called filters)
> - `(3, 3)` = each filter looks at a tiny 3×3 patch of the spectrogram
> - `activation='relu'` = "Rectified Linear Unit" — sets any negative value to 0
>   (adds non-linearity so the network can learn complex patterns)
> - `padding='same'` = keeps the output the same size as the input
> - `input_shape=(128, 128, 1)` = tells Keras the expected input shape

```python
        MaxPooling2D((2, 2)),
```
> Reduces the image by **half** in each dimension (takes the max value in each 2×2 region).
> This makes the model faster and more robust to small shifts in the spectrogram.

### Block 2

```python
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
```
> A deeper convolutional layer learning **64 more complex patterns**.
> Followed by another pooling to halve the size again.

### Classifier Head

```python
        Flatten(),
```
> Converts the 2-D feature map (still shaped like an image) into a simple **1-D list** of numbers.

```python
        Dense(64, activation='relu'),
```
> A standard neural network layer with 64 neurons.
> Every neuron is connected to every input from the Flatten layer.

```python
        Dropout(0.3),
```
> During **training only**, randomly turn off 30% of neurons each step.
> This prevents the model from memorising the training data (overfitting).

```python
        Dense(num_classes, activation='softmax')
```
> The final output layer with one neuron per speaker.
> `softmax` converts raw scores into **probabilities that add up to 1.0**.
> e.g. `[0.80, 0.15, 0.05]` → 80% confident it's speaker 0.

```python
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
```
> Configures how the model learns:
> - `adam` = an intelligent optimizer that adapts the learning speed automatically
> - `categorical_crossentropy` = the standard loss function for multi-class problems
> - `metrics=['accuracy']` = also report accuracy during training so we can watch progress

---

## 🚀 Section 6 — `main()` Function (Lines 111–164)

```python
X_mel, X_pitch, labels = load_dataset()
```
> Calls our loading function to get all features and labels.

```python
le = LabelEncoder()
y_encoded = le.fit_transform(labels)
num_classes = len(le.classes_)
```
> Converts speaker names → integers.
> `fit_transform` learns the mapping AND applies it in one step.
> e.g. `"Rishav_Apple"→0`, `"Utkrarsh_apple"→1`, `"sawant_apple"→2`.

```python
y_cat = to_categorical(y_encoded, num_classes=num_classes)
```
> Converts integers → one-hot vectors.
> e.g. `0 → [1,0,0]`, `1 → [0,1,0]`, `2 → [0,0,1]`.

```python
X_mel_train, X_mel_test, y_train, y_test = train_test_split(
    X_mel, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
)
```
> Splits data: **80% for training**, **20% for testing**.
> - `random_state=42` makes the split reproducible (same result every run)
> - `stratify=y_encoded` ensures each speaker is equally represented in both splits

```python
model = build_cnn(num_classes)
model.summary()
```
> Creates the CNN and prints a table showing each layer's shape and parameter count.

```python
model.fit(X_mel_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_data=(X_mel_test, y_test), verbose=1)
```
> **Trains the model!** This is where all the learning happens.
> - `X_mel_train` = spectrogram images to learn from
> - `y_train` = correct answers (one-hot speaker labels)
> - `validation_data` = data to check accuracy on after each epoch (not used for learning)

```python
loss, acc = model.evaluate(X_mel_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc * 100:.1f}%")
```
> Final evaluation on the unseen test set. Reports accuracy as a percentage.

```python
model.save(MODEL_PATH)
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(le, f)
```
> Saves the trained model to `speaker_model.h5` and the label encoder to `label_encoder.pkl`.
> Both files are needed later by `live_recognition.py`.

---

## 🔄 Full Flow Summary

```
Dataset/
 └── Rishav_Apple/   (5 WAVs)      ┐
 └── Utkrarsh_apple/ (5 WAVs)      ├─ load_dataset() ──► extract_features()
 └── sawant_apple/   (5 WAVs)      ┘         │
                                             ▼
                             X_mel (spectrograms) + labels
                                             │
                             LabelEncoder + train_test_split
                                             │
                                       build_cnn()
                                             │
                                       model.fit()   ← learning happens here
                                             │
                                    model.save() + pickle.dump()
                                             │
                             speaker_model.h5 + label_encoder.pkl
```
