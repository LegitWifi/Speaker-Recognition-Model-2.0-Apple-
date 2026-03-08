# 📗 live_recognition.py — Beginner's Line-by-Line Guide

This document explains **every single line** of `live_recognition.py` in plain English.
Run `train_model.py` first — this script needs the trained model it produces.

---

## 🗺️ Big Picture (What the whole file does)

```
Load saved model  →  Record microphone  →  Extract spectrogram  →  Predict speaker  →  Print result
```

The script lets you speak into your microphone, then uses the trained CNN to tell you
**which speaker** you sound like (and how confident it is).

---

## 📦 Section 1 — Imports (Lines 1–8)

```python
import os
```
> Built-in module for file/folder operations (checking if a file exists, deleting temp files).

```python
import pickle
```
> Used to **load** the label encoder that was saved by `train_model.py`.

```python
import threading
```
> Allows running tasks at the same time (concurrently).
> Imported here for potential use, though the Recorder class uses `sounddevice`'s own threading.

```python
import numpy as np
```
> Fast numerical computation library. Used to manipulate audio arrays.

```python
import librosa
```
> Audio analysis library — used to compute the Mel spectrogram from the recording,
> **exactly the same way** as in `train_model.py`.

```python
import sounddevice as sd
```
> Lets Python access your computer's **microphone and speakers**.
> It captures a live audio stream in real time.

```python
from scipy.io.wavfile import write as wav_write
```
> Saves a NumPy audio array to a `.wav` file on disk.
> We rename it `wav_write` so the name doesn't clash with Python's built-in `write`.

```python
from tensorflow.keras.models import load_model
```
> Loads a previously saved Keras model from an `.h5` file.

---

## ⚙️ Section 2 — Configuration Constants (Lines 11–16)

```python
MODEL_PATH   = "speaker_model.h5"
ENCODER_PATH = "label_encoder.pkl"
```
> The filenames of the model and label encoder saved by `train_model.py`.
> These must exist before running this script.

```python
SAMPLE_RATE  = 22050
N_MELS       = 128
FIXED_WIDTH  = 128
```
> **Must match `train_model.py` exactly.**
> The model was trained on spectrograms of this exact shape — if we change these,
> the features will look different to the model and predictions will be wrong.

```python
TEMP_WAV     = "temp_recording.wav"
```
> A temporary file name used to save the microphone recording before processing it.
> It is deleted automatically after prediction.

---

## 🔬 Section 3 — `extract_features()` Function (Lines 20–39)

This is **identical** to the same function in `train_model.py`.
It is repeated here so this script is self-contained and doesn't depend on the other file.

```python
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
```
> Loads the WAV file into a NumPy array `y` at 22,050 Hz.

```python
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
```
> Computes the Mel spectrogram and converts it to decibel scale —
> the same "sound picture" format the model was trained on.

```python
    if mel_db.shape[1] < FIXED_WIDTH:
        pad_width = FIXED_WIDTH - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :FIXED_WIDTH]
```
> Pad (add zeros) or trim so the spectrogram is always exactly 128 columns wide.

```python
    mel_db = mel_db[:, :, np.newaxis]   # (128, 128, 1)
    return mel_db
```
> Adds the channel dimension and returns the feature array.
> Shape `(128, 128, 1)` matches what the CNN expects.

---

## 🎙️ Section 4 — `Recorder` Class (Lines 42–72)

This class handles all microphone recording.

```python
class Recorder:
    """Handles non-blocking microphone recording."""
```
> A **class** is like a blueprint for an object.
> "Non-blocking" means recording runs in the background — the program
> can do other things (like wait for user input) while audio is being captured.

```python
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.recording = False
        self.frames = []
```
> This is the **constructor** — it runs automatically when you create a `Recorder` object.
> - `self.sample_rate` stores the rate (22,050 Hz)
> - `self.recording` is a flag: `True` = currently recording, `False` = stopped
> - `self.frames` is a list that will collect chunks of audio data

```python
    def _callback(self, indata, frames, time, status):
        if self.recording:
            self.frames.append(indata.copy())
```
> `sounddevice` calls this function **automatically** every time a new chunk of audio
> arrives from the microphone (many times per second).
> - `indata` = the new chunk of audio samples
> - We only save it if `self.recording` is `True`
> - `.copy()` is important — without it, `indata` would be overwritten before we can use it

```python
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
```
> Starts recording:
> 1. Clears any old audio from a previous recording
> 2. Sets the flag to `True` so `_callback` starts saving audio
> 3. Opens a microphone stream with:
>    - `channels=1` = mono audio (single microphone channel)
>    - `dtype='float32'` = each sample is a 32-bit floating point number
>    - `callback=self._callback` = use our function above to receive audio chunks
> 4. Actually starts the stream

```python
    def stop(self):
        self.recording = False
        self.stream.stop()
        self.stream.close()
        if len(self.frames) == 0:
            return None
        audio = np.concatenate(self.frames, axis=0)
        return audio
```
> Stops recording:
> 1. Sets flag to `False` → `_callback` stops saving new audio
> 2. Stops and closes the microphone stream (releases the hardware)
> 3. If no audio was captured (edge case), return `None`
> 4. `np.concatenate` joins all the small chunks in `self.frames` into
>    one big continuous array — the full recording

---

## 💾 Section 5 — `save_wav()` Function (Lines 75–77)

```python
def save_wav(audio, path, sample_rate):
    audio_int = (audio * 32767).astype(np.int16)
    wav_write(path, sample_rate, audio_int)
```
> Saves the audio array to a `.wav` file.
>
> **Why multiply by 32767?**
> - `sounddevice` records audio as float32 values between **-1.0 and +1.0**
> - WAV files store audio as 16-bit integers between **-32768 and +32767**
> - Multiplying by 32767 scales our floats into the integer range
> - `.astype(np.int16)` converts the data type to 16-bit integer

---

## 🔮 Section 6 — `predict_speaker()` Function (Lines 80–88)

```python
def predict_speaker(model, label_encoder, file_path):
```
> Takes the loaded model, the encoder, and a path to a WAV file.
> Returns the predicted speaker name and confidence.

```python
    mel = extract_features(file_path)
```
> Converts the WAV file into a Mel spectrogram — the same format the model was trained on.

```python
    mel_input = np.expand_dims(mel, axis=0)   # (1, 128, 128, 1)
```
> The model expects a **batch** of inputs, shaped `(batch_size, height, width, channels)`.
> Our single sample is `(128, 128, 1)` — we add a batch dimension to make it `(1, 128, 128, 1)`.
> `np.expand_dims(..., axis=0)` inserts a new dimension at position 0.

```python
    predictions = model.predict(mel_input, verbose=0)
```
> Runs the spectrogram through the trained CNN.
> Returns a 2-D array of probabilities, e.g. `[[0.80, 0.15, 0.05]]`
> (one value per speaker, all adding up to 1.0).

```python
    predicted_class = np.argmax(predictions)
```
> `np.argmax` finds the **index of the highest value**.
> In `[0.80, 0.15, 0.05]`, the max is at index `0` → predicted class is `0`.

```python
    speaker_name = label_encoder.inverse_transform([predicted_class])[0]
```
> Converts the integer class index back to the human-readable speaker name.
> e.g. `0 → "Rishav_Apple"`.

```python
    confidence = predictions[0][predicted_class] * 100
    return speaker_name, confidence
```
> Converts the probability to a percentage (e.g. `0.80 → 80.0`).
> Returns both the name and the confidence score.

---

## 🎬 Section 7 — `main()` Function (Lines 91–157)

### Loading the Model

```python
if not os.path.exists(MODEL_PATH):
    print(f"\n[ERROR] Model not found: '{MODEL_PATH}'")
    print("  Please run  train_model.py  first.")
    return
```
> Checks if the model file exists before trying to load it.
> If not, prints a helpful error and exits cleanly instead of crashing.

```python
model = load_model(MODEL_PATH)
```
> Loads the trained CNN from `speaker_model.h5` back into memory.

```python
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)
```
> Opens `label_encoder.pkl` in **read-binary** mode (`"rb"`) and loads the encoder.
> The `with` statement ensures the file is automatically closed after loading.

```python
print(f"\nKnown speakers: {list(label_encoder.classes_)}")
```
> Prints the list of speakers the model knows about.
> e.g. `['Rishav_Apple', 'Utkrarsh_apple', 'sawant_apple']`

```python
recorder = Recorder(SAMPLE_RATE)
```
> Creates a `Recorder` object, ready to capture microphone input.

### Main Loop

```python
while True:
    user_input = input("\nType 'start' to record  |  'stop' to exit: ").strip().lower()
```
> Runs forever until the user types `"stop"`.
> `.strip()` removes accidental spaces. `.lower()` makes it case-insensitive.

```python
    if user_input == "stop":
        print("Goodbye!")
        break
```
> If the user types `"stop"` at the main menu, exit the loop (and the program).

```python
    if user_input != "start":
        print("  → Please type 'start' to begin.")
        continue
```
> If the user types anything other than `"start"` or `"stop"`, remind them and loop again.

### Recording Phase

```python
    print("\n🎙  Recording...  (type 'stop' and press Enter to finish)")
    recorder.start()
```
> Prints a message and starts the microphone stream.
> Audio is now being captured in the background.

```python
    stop_input = input("").strip().lower()
    while stop_input != "stop":
        stop_input = input("  → Type 'stop' to stop recording: ").strip().lower()
```
> Waits for the user to type `"stop"`.
> While the program is waiting here, the `Recorder._callback` is still running in the
> background, continuously saving microphone audio to `self.frames`.

```python
    audio = recorder.stop()
```
> Stops recording and returns the complete audio as a NumPy array.

```python
    if audio is None or len(audio) == 0:
        print("[!] No audio captured. Please try again.")
        continue
```
> Safety check — if nothing was captured (e.g. microphone not connected), skip prediction.

```python
    duration = len(audio) / SAMPLE_RATE
    print(f"\n[✓] Captured {duration:.1f} seconds of audio.")
```
> Calculates how many seconds were recorded:
> `number of samples ÷ samples per second = seconds`

```python
    save_wav(audio, TEMP_WAV, SAMPLE_RATE)
    print("[✓] Processing audio...")
```
> Saves the recording to a temporary WAV file.
> `librosa.load()` (used inside `extract_features`) works with files, not raw arrays,
> so we need this intermediate step.

```python
    speaker, confidence = predict_speaker(model, label_encoder, TEMP_WAV)
```
> Runs the full prediction pipeline on the saved WAV file.

```python
    print(f"  ✅  Matched Speaker : {speaker}")
    print(f"  📊  Confidence      : {confidence:.1f}%")
```
> Prints the result!
> `:.1f` formats the number to 1 decimal place (e.g. `80.0%`).

```python
    if os.path.exists(TEMP_WAV):
        os.remove(TEMP_WAV)
```
> Deletes the temporary WAV file to keep the folder clean.
> `os.path.exists` check prevents an error if the file was already removed somehow.

---

## 🔄 Full Flow Summary

```
python live_recognition.py
        │
        ▼
load speaker_model.h5  +  label_encoder.pkl
        │
        ▼  (loop)
User types "start"
        │
        ▼
Microphone starts recording  (background thread)
        │
        ▼
User types "stop"
        │
        ▼
Audio array  →  save to temp_recording.wav
        │
        ▼
extract_features()  →  Mel spectrogram (128×128×1)
        │
        ▼
model.predict()  →  [0.80, 0.15, 0.05]
        │
        ▼
argmax → class index → label_encoder → "Rishav_Apple"
        │
        ▼
Print "Matched Speaker: Rishav_Apple  |  Confidence: 80.0%"
        │
        ▼
Delete temp WAV  →  loop back to "start"
```

---

## 🔑 Key Terms Glossary

| Term | Plain-English Meaning |
|---|---|
| **Spectrogram** | A "photo" of sound showing frequency vs time |
| **Mel scale** | Frequency spacing that matches human hearing |
| **CNN** | Convolutional Neural Network — works like a pattern-detector for images |
| **Epoch** | One full pass of the training data through the model |
| **Softmax** | Converts raw scores into probabilities that add up to 100% |
| **Label Encoder** | Translates between speaker names (text) and numbers |
| **Dropout** | Randomly disables neurons during training to prevent overfitting |
| **Batch** | A small group of samples processed together before updating the model |
| **Inference** | Using a trained model to make a prediction (the opposite of training) |
