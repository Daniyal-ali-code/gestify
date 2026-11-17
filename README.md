# Gestify — ASL Fingerspelling Recognition

**Turning Hand Gestures into Words.**

Gestify is a Python desktop application that uses a webcam, MediaPipe
hand tracking and a pretrained TensorFlow Lite classifier to recognize
American Sign Language (ASL) hand gestures and fingerspelled letters.

It can display recognized gestures and letters and compose recognized
output into text.

This is an actively developed experimental prototype, not a complete
ASL language translation system.

---

# 1. Current Project Status

**Status: Working webcam prototype; recognition accuracy testing pending.**

The application has successfully launched on the development laptop.
Live hand tracking, gesture recognition and text/word output have been
manually tested and reported working.

Verified technical results:

- Webcam successfully opens using OpenCV camera index 0.
- Webcam frame resolution: 640x480.
- MediaPipe 0.10.14 initializes without the previous CPU AES crash.
- Gesture Recognizer model initializes successfully.
- MediaPipe successfully processes real webcam frames.
- A captured image containing a hand produced one detected hand.
- The detected hand returned 21 landmarks.
- Landmark coordinates have shape (21, 3).
- ASL TensorFlow Lite classifier initializes successfully.
- Application setup check passes.
- Live application was manually tested successfully.

**Important:** These results confirm that the application and tracking
pipeline work. They do not establish a measured recognition accuracy
for every ASL letter.

---

# 2. Development Environment

The application was developed and tested on:

| Component | Tested configuration |
|---|---|
| OS | Fedora Linux |
| Python | 3.11.18 |
| CPU | Intel Core i3 M 330 @ 2.13 GHz |
| CPU architecture | x86_64 |
| CPU AES support | Not advertised in lscpu flags |
| Integrated GPU | Intel HD Graphics (Ironlake) |
| Tk | 8.6 |
| Camera | /dev/video0 |
| Working camera index | 0 |
| Working resolution | 640x480 |

The laptop is an older, low-resource machine. CPU compatibility and
memory usage are important design constraints.

The application should not require a modern GPU.

# 3. Main Technologies and Tested Versions

The working development environment contains:

| Package | Tested version |
|---|---|
| MediaPipe | 0.10.14 |
| NumPy | 2.4.6 |
| OpenCV contrib | 5.0.0.93 |
| Protobuf | 4.25.8 |
| attrs | 26.1.0 |
| LiteRT | 2.3.0 |
| CustomTkinter | 6.0.0 |
| Pillow | 13.0.0 |
| Python | 3.11.18 |

See requirements.txt for the pinned base dependencies.

**Critical CPU compatibility note:**

MediaPipe 1.2.3 caused an illegal-instruction crash while initializing
the tracking model on the development laptop.

The error was:

    FATAL ERROR: This binary was compiled with aes enabled,
    but this feature is not available on this processor.

The Intel Core i3 M330 does not advertise AES support.

MediaPipe 0.10.14 successfully initializes the Gesture Recognizer
on this same processor. Keep MediaPipe pinned to 0.10.14 on this
tested machine. Do not casually upgrade it to a newer version.

MediaPipe is installed separately with --no-deps because its unused
JAX/JAXLib dependencies are not required by this application's
MediaPipe Tasks tracking pipeline.

---

# 4. Features and Application Modes

## Alphabet mode

- Uses a pretrained ASL alphabet keypoint classifier.
- Includes labels A-Z.
- Uses hand landmarks as classifier input.
- Recognizes static hand configurations.
- Recognized letters can be composed into text.

**J and Z limitation:**

ASL J and Z require movement. The current alphabet classifier
classifies individual frames rather than full motion sequences.
Therefore J and Z are unreliable.

The other 24 static letters are intended to be recognized, but their
accuracy has not yet been formally benchmarked.

## Gesture mode

Uses MediaPipe's pretrained Gesture Recognizer.

The built-in gesture mappings include:

- Open_Palm -> HELLO
- Thumb_Up -> YES
- Thumb_Down -> NO
- Closed_Fist -> STOP
- Pointing_Up -> ATTENTION
- Victory -> PEACE
- ILoveYou -> I LOVE YOU

These mappings can be changed in Settings.

## Custom Hold

Allows the user to teach the application held hand-shape gestures.

## Custom Motion

Allows the user to teach the application movement-based gestures.

## Text composition

The application supports composing recognized output into words
and displaying the resulting text.

---

# 5. Model Files

All three model files are currently included in the models directory.

| File | Purpose |
|---|---|
| gesture_recognizer.task | Gesture recognition and primary hand tracking |
| hand_landmarker.task | Fallback hand landmark tracking |
| keypoint_classifier.tflite | ASL alphabet A-Z classifier |

Approximate file sizes:

- gesture_recognizer.task: 8.0 MB
- hand_landmarker.task: 7.5 MB
- keypoint_classifier.tflite: 28 KB

## Model sources

MediaPipe Gesture Recognizer:

https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task

MediaPipe Hand Landmarker:

https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

ASL alphabet classifier:

https://raw.githubusercontent.com/Muhib-Mehdi/ASL-Recognition-System/main/model/keypoint_classifier/keypoint_classifier.tflite

Credits:

- MediaPipe by Google.
- ASL alphabet keypoint classifier by Muhib-Mehdi (MIT).
- The initial Ars Signorum application foundation was obtained from
  SdheroPro/ai-sign-language-translator. The original MIT license
  is retained.

---

# 6. Troubleshooting History — Issues and Fixes

This section records the actual debugging process so future
development sessions do not repeat unsuccessful investigations.

## Issue 1: Missing Tkinter

Initial error:

    Tk is missing from this Python.

Cause:

The Python 3.11 installation did not have Tkinter available.

Fix:

Installed the Fedora Tkinter package:

    sudo dnf install python3.11-tkinter

Verification:

    python -c "import tkinter; print(tkinter.TkVersion)"

Result:

    Tkinter OK: 8.6

Status: FIXED.

## Issue 2: Missing hand-tracking models

Initial application message:

    No hand tracking model - open models tab

The Models tab displayed:

    Download failed - Paste file manually

Investigation showed:

- gesture_recognizer.part
- hand_landmarker.part

These were incomplete downloads.

The official MediaPipe model files were downloaded manually and
saved with their expected filenames:

    models/gesture_recognizer.task
    models/hand_landmarker.task

The incomplete .part files were removed.

The ASL classifier was already bundled:

    models/keypoint_classifier.tflite

Status: FIXED.

## Issue 3: Illegal instruction / AES CPU incompatibility

Initial error:

    FATAL ERROR: This binary was compiled with aes enabled,
    but this feature is not available on this processor.

The installed MediaPipe 1.2.3 imported successfully, but both
Gesture Recognizer and Hand Landmarker initialization caused
the crash.

The CPU flags showed that Intel Core i3 M330 does not advertise AES.

The ASL LiteRT classifier could initialize independently, so
LiteRT was not identified as the cause of this particular crash.

Fix:

MediaPipe was downgraded to 0.10.14.

The older wheel was recovered from the local pip cache, avoiding
another large internet download.

Protobuf 4.25.8 and attrs 26.1.0 were also installed from cache.

MediaPipe was installed with:

    python -m pip install --no-index --no-deps \
        /path/to/mediapipe-0.10.14.whl

After installation:

    python -c "import mediapipe as mp; print(mp.__version__)"

Result:

    0.10.14

The Gesture Recognizer then initialized successfully without
the AES crash.

Status: FIXED ON THE TESTED CPU.

Do not upgrade MediaPipe on this laptop without retesting CPU
compatibility.

## Issue 4: Slow package downloads

Large downloads from PyPI and an alternative mirror were
extremely slow and repeatedly interrupted.

The attempted MediaPipe 0.10.21 installation was abandoned.

Instead of repeatedly downloading packages, the local pip HTTP
cache was inspected and used to recover compatible wheels.

Useful diagnostic command:

    python -m pip cache list

Note: pip cache list only displays its wheel cache. Downloaded
HTTP cache bodies can contain wheels that are not displayed by
that command.

No reason exists to repeat the large downgrade downloads when
the working MediaPipe 0.10.14 environment is already available.

Status: RESOLVED FOR CURRENT DEVELOPMENT ENVIRONMENT.

## Issue 5: Camera index errors

Initial messages included:

    Camera index out of range

Fedora exposed:

    /dev/video0
    /dev/video1

Independent OpenCV testing showed:

- Camera 0 successfully captured a 640x480 frame.
- Camera 1 could not be opened.

Fix:

Use camera index 0.

The default resolution was reduced from 1280x720 to 640x480
to reduce processing load on the old laptop.

Status: FIXED.

## Issue 6: Zero hand detections in an initial test

An early 30-frame test returned:

    Frames tested: 30
    Frames with detected hands: 0
    Maximum hands detected: 0

This initially appeared to indicate a detection failure.

However, the user was not showing a hand to the camera during
that test. Therefore the result was inconclusive.

A live camera capture was then performed with the user showing
an open palm and pressing S to save the image.

The image was saved as camera_debug.jpg.

MediaPipe was tested against that actual image.

Result:

    Hands detected: 1
    Hand 1: 21 landmarks
    Coordinates: (21, 3)

Status: RESOLVED; the original zero-detection result was caused
by testing without a hand in view.

## Issue 7: Live camera and MediaPipe integration

After fixing MediaPipe compatibility and identifying camera 0,
a real webcam frame was successfully passed through the
Gesture Recognizer.

Verification confirmed:

- Camera frame: (480, 640, 3)
- Tracker initialization: successful
- MediaPipe frame processing: successful

The complete desktop application was then launched.

The user confirmed that live tracking worked very well and that
the application could detect signs and display recognized words.

Status: WORKING IN MANUAL TESTING.

---

# 7. Important Diagnostic Commands

Activate the environment:

    cd ~/Projects/gestify
    source .venv/bin/activate

Check the environment:

    python app.py --check

Compile the Python files:

    python -m py_compile app.py sign_language_setup.py

Launch the application:

    ./run.sh

Inspect CPU features:

    lscpu | grep -E 'Model name|Flags'

Check available cameras:

    ls -l /dev/video*

Check installed relevant packages:

    python -m pip show mediapipe numpy opencv-contrib-python protobuf ai-edge-litert

## Camera configuration

The tested settings are:

    camera_index: 0
    resolution: 640x480
    show_landmarks: true
    mode: Alphabet

The application stores these settings in settings.json.

Do not commit personal runtime data or webcam debug images.

---

# 8. Installation on Fedora

The tested Python version is Python 3.11.

Install Tkinter:

    sudo dnf install python3.11-tkinter

Create a virtual environment:

    python3.11 -m venv .venv

Activate it:

    source .venv/bin/activate

Install the pinned base dependencies:

    python -m pip install -r requirements.txt

Install the tested MediaPipe version without pulling unused
JAX dependencies:

    python -m pip install --no-deps mediapipe==0.10.14

Check the setup:

    python app.py --check

Run:

    ./run.sh

The bundled setup script is also available:

    python sign_language_setup.py --dir . --no-run

Then:

    ./run.sh

The initial installation needs internet access for packages and
any model files that are not already present.

---

# 9. Repository and Git Workflow

GitHub repository:

https://github.com/Daniyal-ali-code/gestify

Default branch:

    main

Important local directories:

    ~/Projects/gestify
    ~/Projects/gestify-publish
    ~/Projects/signbridge-ai

The active development project is:

    ~/Projects/gestify

The publishing clone is:

    ~/Projects/gestify-publish

The publishing clone's origin points to:

    https://github.com/Daniyal-ali-code/gestify.git

**Important Git warning:**

The development directory ~/Projects/gestify was originally cloned
from SdheroPro/ai-sign-language-translator. Its Git remote may still
point to that upstream repository.

Use ~/Projects/gestify-publish for commits and pushes to the user's
Daniyal-ali-code/gestify repository unless its remote is deliberately
changed and verified.

The separate ~/Projects/signbridge-ai directory was not modified
during the Gestify publishing process.

The virtual environment, __pycache__, generated data and temporary
downloads should not be committed.

The models directory is intentionally committed because the project
requires these pretrained models.

---

# 10. Current Limitations

1. The 24 static ASL letters have not undergone a formal accuracy test.
2. J and Z require motion recognition and are unreliable with the
   current single-frame alphabet classifier.
3. Performance and latency should be measured on the older CPU.
4. Recognition should be tested under different lighting conditions,
   hand distances and camera angles.
5. Gesture recognition and fingerspelling do not amount to full
   continuous ASL sentence translation.
6. The fallback Hand Landmarker has not been independently
   revalidated after the MediaPipe downgrade.

Do not describe the application as a fully accurate ASL translator
until the relevant recognition tests have been completed.

---

# 11. Next Development Tasks

Continue from this list:

## Priority 1 — Alphabet recognition evaluation

- Test all static letters A-I and K-Y.
- Record correct and incorrect predictions.
- Test repeated recognition of each letter.
- Measure false predictions and recognition stability.
- Adjust confidence thresholds and stabilization only after
  collecting test results.

## Priority 2 — J and Z motion recognition

- Capture sequences of hand landmarks across multiple frames.
- Develop a lightweight motion classifier or trajectory rules.
- Keep CPU and memory requirements low.
- Test J and Z separately from static letters.

## Priority 3 — Live performance

- Measure FPS on the Intel Core i3 M330.
- Check CPU and memory usage.
- Reduce unnecessary camera processing.
- Maintain a responsive desktop interface.

## Priority 4 — Testing and documentation

- Test with different lighting conditions and hand positions.
- Record recognition results.
- Keep this README updated with verified results.
- Avoid claiming accuracy without measured evidence.

---

# 12. Handoff Notes for Future Development Sessions

When continuing this project in a new development session:

1. Read this README first.
2. Use Python 3.11.
3. Preserve MediaPipe 0.10.14 on the tested old CPU.
4. Use camera 0 at 640x480.
5. Work in ~/Projects/gestify for application development.
6. Use ~/Projects/gestify-publish for GitHub commits and pushes
   until the development remote has been corrected.
7. Do not repeat the already-solved Tkinter, missing-model,
   AES compatibility or camera-index investigations.
8. Start with alphabet recognition evaluation, followed by J/Z
   motion recognition.

The immediate next task is to benchmark the 24 static ASL letters.
