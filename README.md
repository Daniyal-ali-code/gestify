# Gestify

Gestify is an experimental desktop application for webcam-based ASL
fingerspelling and hand-gesture recognition.

## Current development status

The application launches, loads its bundled ASL classifier, initializes
MediaPipe 0.10.14, and successfully processes webcam frames.

Actual hand detection is still under investigation. A recent 30-frame
test returned zero detected hands, so recognition is not yet considered
fully operational.

## Features

- Alphabet mode with a bundled ASL A-Z keypoint classifier.
- Gesture mode with MediaPipe's pretrained gesture recognizer.
- Custom Hold and Custom Motion gesture modes.
- Text composition and configurable recognition settings.

The alphabet classifier processes static hand shapes. J and Z require
movement and are not reliably recognized by this static classifier.

## Requirements

- Python 3.11
- A webcam
- Tkinter
- The pinned dependencies in requirements.txt

The tested MediaPipe version is 0.10.14. It is installed separately
without its unused JAX dependencies to avoid unnecessary downloads.
Do not upgrade MediaPipe automatically to a newer major version on the
tested older CPU configuration.

## Fedora installation

Install Tkinter:

    sudo dnf install python3.11-tkinter

Create the environment:

    python3.11 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    python -m pip install -r requirements.txt
    python -m pip install --no-deps mediapipe==0.10.14

Launch:

    ./run.sh

Alternatively, use the included setup script:

    python sign_language_setup.py --dir . --no-run
    ./run.sh

## Bundled models

- models/gesture_recognizer.task
- models/hand_landmarker.task
- models/keypoint_classifier.tflite

The models are included in this repository.

## Limitations

- J and Z require dynamic movement recognition.
- Real-world recognition accuracy is still being tested.
- This is an experimental prototype, not a complete ASL translation system.

## Credits

- MediaPipe by Google.
- ASL alphabet keypoint classifier by Muhib-Mehdi (MIT).
