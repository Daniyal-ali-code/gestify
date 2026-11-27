# Gestify

**Real-time hand-gesture and ASL fingerspelling recognition on your desktop.**

Gestify is a Python desktop application that uses a webcam and hand-landmark tracking to recognize selected hand gestures, classify ASL fingerspelling letters, and let users train their own static and movement-based gestures.

It is an experimental recognition and learning tool. It is **not** a complete American Sign Language (ASL) translator and does not currently interpret continuous ASL sentences, grammar, or facial expressions.

## Contents

- [Features](#features)
- [Recognition modes](#recognition-modes)
- [Speech-to-Text](#speech-to-text)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running Gestify](#running-gestify)
- [Using the application](#using-the-application)
- [Project files and local data](#project-files-and-local-data)
- [Models and acknowledgements](#models-and-acknowledgements)
- [Troubleshooting](#troubleshooting)
- [Current limitations](#current-limitations)
- [Development roadmap](#development-roadmap)
- [Developers](#developers)

## Features

- Live webcam preview with hand tracking.
- Four recognition modes: Gestures, Alphabet, Custom Hold, and Custom Motion.
- ASL alphabet classification for A–Z using a pretrained keypoint classifier.
- Configurable mappings from built-in gestures to words or phrases.
- In-app tools for collecting and training custom static and dynamic gestures.
- Recognition output that can be composed into text.
- Recognition history and a model library.
- English-only offline Speech-to-Text with live captions.
- Microphone controls and transcript copy, clear, and save options.
- Dark and light appearance options.
- Adjustable camera, detection, confidence, and recognition settings.
- CPU-oriented operation; a dedicated GPU is not required by the documented setup.

## Recognition modes

### 1. Gestures

Uses MediaPipe's pretrained Gesture Recognizer. The application maps recognized categories to configurable output text.

Default mappings include:

| Detected gesture | Default output |
|---|---|
| Open_Palm | HELLO |
| Thumb_Up | YES |
| Thumb_Down | NO |
| Closed_Fist | STOP |
| Pointing_Up | ATTENTION |
| Victory | PEACE |

The output mappings can be changed in the application settings. These are application-defined gesture-to-text mappings, not translations of complete ASL expressions.

### 2. Alphabet

Uses a pretrained TensorFlow Lite keypoint classifier to classify hand configurations as ASL letters A–Z. The classifier receives hand-landmark data rather than a full image.

**Important:** J and Z are movement-based letters. A classifier that evaluates individual frames cannot reliably recognize their complete movement paths. The other 24 static letters still require systematic accuracy testing; no formal accuracy percentage is claimed.

### 3. Custom Hold

Lets a user collect examples of a held hand shape, assign it a label, and train a custom static gesture recognizer using the in-app trainer.

### 4. Custom Motion

Lets a user collect examples of a gesture that changes over time and train a custom movement-based recognizer. Recognition uses a sequence of hand-landmark observations rather than a single static pose.

Custom recognition quality depends on the number, consistency, and variety of examples collected. Train and test custom gestures under conditions similar to real use.

## Speech-to-Text

Gestify includes an English-only Speech-to-Text page that converts microphone
speech into on-screen captions.

Features:

- Live microphone capture with Start Listening and Stop controls.
- English speech recognition using the Whisper Base model.
- Transcript displayed inside the application.
- Copy, clear, and save transcript options.
- Speech processing runs locally on the computer. Audio is not sent to a
  cloud transcription service, and no API key or account is required.

### First-time model setup

The Whisper Base model must be available locally before using Speech-to-Text.
Download it once while connected to the internet:

```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8'); print('Whisper Base model is ready.')"
```

This downloads the model into the local model cache. After the download,
Gestify loads it in local-only mode, so Speech-to-Text can operate without an
internet connection.

On Windows, run the command using the project virtual environment if needed:

```powershell
.\.venv\Scripts\python.exe -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8'); print('Whisper Base model is ready.')"
```

The first model load can take some time, especially on a CPU. Recognition speed
and accuracy depend on the computer, microphone, background noise, and speech
clarity. Captions are produced in chunks, so they may not appear word-by-word
with zero delay. The feature is intended for English speech and does not claim
perfect transcription accuracy.

## How it works

The main recognition pipeline is:

1. Open a webcam and capture a frame.
2. Process the frame with the available MediaPipe task model.
3. Detect a hand and obtain its landmarks.
4. Send the landmark data to the selected recognizer.
5. Apply the configured confidence and recognition-stability settings.
6. Display the recognized label and, where applicable, add it to the text output.

The selected mode determines which recognizer processes the hand data. Gesture recognition uses the pretrained MediaPipe Gesture Recognizer; Alphabet uses the ASL keypoint classifier; Custom Hold and Custom Motion use user-collected training samples.

## Requirements

- Python 3.11.
- A working webcam for sign recognition.
- A working microphone for Speech-to-Text.
- Windows or Linux.
- A supported Tk installation (Tkinter is included with many Windows Python installations; on Fedora it may need to be installed separately).
- Internet access for initial dependency or model downloads, if the required files are not already present.

### Tested development environment

The project has been tested in these environments:

| Component | Tested configuration |
|---|---|
| Windows Python | 3.11.9 |
| Fedora Python | 3.11.18 |
| MediaPipe | 0.10.14 |
| NumPy | 2.4.6 |
| OpenCV contrib | 5.0.0.93 |
| CustomTkinter | 6.0.0 |
| faster-whisper | 1.2.1 |
| sounddevice | 0.5.6 |
| LiteRT | 2.3.0 |
| Camera index | 0 |
| Camera resolution | 640 × 480 |

The tested Fedora laptop uses an older Intel Core i3 M330 CPU. MediaPipe 0.10.14 initialized successfully on that machine. A newer MediaPipe build previously caused an AES-instruction compatibility crash on that particular CPU. Keep the tested version on that hardware unless a newer version has been verified there.

The exact packages pinned in [requirements.txt](requirements.txt) are the project's base Python dependencies. MediaPipe 0.10.14 is installed separately because the tested setup uses it without installing its optional dependency set.

## Installation

### Windows (PowerShell)

Clone the repository and enter its directory:

```powershell
git clone https://github.com/Daniyal-ali-code/gestify.git
cd gestify
```

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the base dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the tested MediaPipe version:

```powershell
python -m pip install --no-deps mediapipe==0.10.14
```

If PowerShell blocks virtual-environment activation, run the interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps mediapipe==0.10.14
```

### Fedora Linux

Install Python 3.11 and Tkinter (package names can vary by Fedora release):

```bash
sudo dnf install python3.11 python3.11-tkinter
```

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps mediapipe==0.10.14
```

If the system does not provide the requested Python package under that exact name, install the equivalent Python 3.11 and Tkinter packages available for your Fedora release.

## Running Gestify

From the project directory:

**Windows**

```powershell
.\run.bat
```

Or:

```powershell
.\.venv\Scripts\python.exe .\app.py
```

**Linux**

```bash
chmod +x run.sh
./run.sh
```

Or:

```bash
.venv/bin/python app.py
```

### Check the installation

Run the application setup check:

```bash
python app.py --check
```

Compile the main Python files:

```bash
python -m py_compile app.py sign_language_setup.py
```

On Windows, use the virtual-environment interpreter if it is not activated:

```powershell
.\.venv\Scripts\python.exe .\app.py --check
.\.venv\Scripts\python.exe -m py_compile .\app.py .\sign_language_setup.py
```

## Using the application

1. Start Gestify using the launcher for your operating system.
2. Allow camera access if the operating system requests it.
3. Select the desired recognition mode.
4. Position your hand clearly inside the camera frame.
5. Use the on-screen controls to start recognition and view the predicted output.
6. For custom recognition, open the Custom Trainer, record examples for each label, and train the model.
7. Review previous recognition activity in History and inspect available models in Model Library.
8. Adjust recognition thresholds, camera options, text formatting, and gesture-to-text mappings in settings when necessary.
9. Open Speech-to-Text, click Start Listening, and speak English into your microphone. Use Stop when finished, then copy or save the transcript.

For more reliable results, use even lighting, keep the hand inside the camera frame, avoid cluttered backgrounds, and collect varied training examples for custom gestures.

## Project files and local data

```text
gestify/
├── app.py                  # Main desktop application
├── sign_language_setup.py  # Setup and dependency/model helper
├── requirements.txt        # Pinned base Python dependencies
├── run.bat                 # Windows launcher
├── run.sh                  # Linux launcher
├── settings.json           # Application preferences
├── models/
│   ├── gesture_recognizer.task
│   ├── hand_landmarker.task
│   └── keypoint_classifier.tflite
├── data/                   # Locally collected samples and history
└── README.md
```

The application may create or use the following local data files:

- `data/static_samples.npz` — samples for custom held gestures.
- `data/dynamic_samples.npz` — samples for custom movement gestures.
- `data/history.log` — recognition history.
- `settings.json` — camera, appearance, mode, and recognition preferences.

Keep personal training samples and local preferences backed up. Do not publish private or sensitive recordings. The `.venv` directory and Python cache files are generated locally and should not be committed.

## Models and acknowledgements

The application uses these model files:

| File | Role | Source |
|---|---|---|
| `gesture_recognizer.task` | Gesture classification and hand tracking | [MediaPipe Gesture Recognizer](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task) |
| `hand_landmarker.task` | Fallback hand-landmark detection | [MediaPipe Hand Landmarker](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) |
| `keypoint_classifier.tflite` | ASL alphabet keypoint classification | [ASL Recognition System model](https://raw.githubusercontent.com/Muhib-Mehdi/ASL-Recognition-System/main/model/keypoint_classifier/keypoint_classifier.tflite) |

Acknowledgements:

- [MediaPipe](https://developers.google.com/mediapipe) by Google provides the hand and gesture task framework and models.
- The ASL alphabet keypoint classifier is by Muhib-Mehdi and is distributed under its MIT license.

Third-party software and model terms remain applicable to their respective components. See the included license and the relevant upstream project terms before redistributing or modifying third-party components.

## Troubleshooting

### Camera does not open

- Check that the webcam is connected and not being used by another application.
- Try camera index 0 first; it is the index used in the tested setup.
- On Linux, inspect available video devices with `ls -l /dev/video*`.
- Try the 640 × 480 resolution to reduce processing load.

### Tkinter is missing

On Fedora, install the Python 3.11 Tkinter package:

```bash
sudo dnf install python3.11-tkinter
```

On Windows, use a Python installation that includes Tcl/Tk support.

### A model is missing or fails to load

- Confirm that the required files exist in the `models/` directory.
- Check that downloads completed and that files are not partial downloads.
- Run `python app.py --check` to inspect the application setup.
- Use the Model Library or setup helper to obtain missing model files.

### MediaPipe crashes on an older CPU

A previous tested configuration using a newer MediaPipe build failed on an Intel Core i3 M330 because the binary expected AES CPU support. MediaPipe 0.10.14 worked on that machine. This is a hardware-specific compatibility observation, not a guarantee for every CPU.

### Recognition is inconsistent

- Improve lighting and keep the hand fully visible.
- Keep the camera steady and avoid rapid unintended movement.
- Review confidence and stabilization settings.
- For custom modes, record more examples from slightly different positions and repeat the training process.
- Do not infer overall ASL accuracy from a few successful predictions; evaluate each letter and gesture systematically.

## Current limitations

- The 24 static ASL letters have not yet been formally benchmarked.
- J and Z require motion-aware recognition and are not reliably handled by the current single-frame alphabet classifier.
- Gesture-to-word mappings are configurable shortcuts, not full ASL translation.
- The application does not currently provide continuous sentence-level ASL translation, ASL grammar interpretation, or facial-expression analysis.
- Recognition quality depends on camera quality, lighting, hand position, and training data.
- Performance and recognition latency can vary with hardware.
- Speech-to-Text currently supports English only. Transcription accuracy is not guaranteed and may be affected by noise, accent, microphone quality, and CPU performance.

## Development roadmap

1. **Evaluate static alphabet recognition:** benchmark A–I and K–Y, record correct and incorrect predictions, and measure stability and false detections.
2. **Improve J and Z recognition:** investigate motion sequences and lightweight trajectory-based classification.
3. **Measure live performance:** record frame rate, CPU use, memory use, and recognition latency on low-resource hardware.
4. **Improve testing:** evaluate different lighting conditions, camera angles, hand distances, and users.
5. **Maintain accurate documentation:** publish measured results and clearly distinguish tested behavior from planned work.

## Developers

- **Daniyal Ali**
- **Ayesha Zafar**

Gestify is an actively developed project. Contributions, reproducible bug reports, and carefully documented recognition tests are welcome.
