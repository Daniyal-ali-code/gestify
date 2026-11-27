#!/usr/bin/env python3
"""Gestify - Sign Language Interpreter.

Camera -> hand tracking (MediaPipe Tasks) -> pretrained recognisers -> words on screen.

Recognition modes
  Gestures       pretrained MediaPipe Gesture Recognizer (8 canned gestures -> your words)
  Alphabet       pretrained ASL A-Z keypoint classifier (TFLite)
  Custom Hold    your own held signs (trained inside the app)
  Custom Motion  your own moving signs (trained inside the app)

Run:  python app.py
"""
import copy
import importlib
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

try:
    import cv2
    import numpy as np
    import customtkinter as ctk
    from PIL import Image
    from tkinter import filedialog
except ImportError as exc:  # pragma: no cover
    name = getattr(exc, "name", None) or str(exc)
    hint = ""
    if "tkinter" in str(exc).lower() or name == "tkinter":
        hint = "\nOn Linux install Tk first:  sudo apt install python3-tk"
    sys.exit(f"Missing package: {name}\nRun setup again (python sign_language_setup.py) or:  "
             f"pip install -r requirements.txt{hint}")

APP_NAME = "Gestify"
APP_SUB = "ASL FINGERSPELLING RECOGNITION"
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
SETTINGS_PATH = BASE / "settings.json"
STATIC_FILE = DATA_DIR / "static_samples.npz"
DYNAMIC_FILE = DATA_DIR / "dynamic_samples.npz"
HISTORY_FILE = DATA_DIR / "history.log"

MODELS = {
    "gesture": {
        "file": "gesture_recognizer.task",
        "title": "Gesture Recognizer",
        "purpose": "Pretrained hand gestures (8 built-in) and hand tracking. Recommended.",
        "url": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
               "gesture_recognizer/float16/1/gesture_recognizer.task",
        "min_size": 1_000_000,
    },
    "asl": {
        "file": "keypoint_classifier.tflite",
        "title": "ASL Alphabet Classifier",
        "purpose": "Pretrained ASL letters A-Z from hand landmarks (MIT, Muhib-Mehdi).",
        "url": "https://raw.githubusercontent.com/Muhib-Mehdi/ASL-Recognition-System/main/"
               "model/keypoint_classifier/keypoint_classifier.tflite",
        "min_size": 10_000,
    },
    "hand": {
        "file": "hand_landmarker.task",
        "title": "Hand Landmarker (fallback)",
        "purpose": "Hand tracking only. Used if the Gesture Recognizer file is missing.",
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
               "hand_landmarker/float16/1/hand_landmarker.task",
        "min_size": 1_000_000,
    },
}
ASL_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MODES = ["Gestures", "Alphabet", "Custom Hold", "Custom Motion"]
CASES = ["UPPERCASE", "lowercase", "Capitalized", "As is"]
RESOLUTIONS = ["640x480", "1280x720", "1920x1080"]

DEFAULTS = {
    "appearance": "Dark",
    "mode": "Gestures",
    "camera_index": 0,
    "resolution": "640x480",
    "mirror": True,
    "show_landmarks": True,
    "show_overlay_text": True,
    "min_detection": 0.6,
    "min_presence": 0.5,
    "min_tracking": 0.5,
    "gesture_min_score": 0.6,
    "alphabet_min_conf": 0.8,
    "hold_frames": 10,
    "cooldown": 1.0,
    "repeat_delay": 2.0,
    "auto_space_seconds": 2.0,
    "knn_k": 5,
    "static_max_distance": 1.6,
    "dynamic_max_distance": 2.2,
    "min_motion": 0.12,
    "dynamic_window": 18,
    "dynamic_resample": 16,
    "text_case": "UPPERCASE",
    "speak_on_word": False,
    "auto_download": True,
    "geometry": "1280x820",
    "gesture_words": {
        "Open_Palm": "HELLO",
        "Thumb_Up": "YES",
        "Thumb_Down": "NO",
        "Closed_Fist": "STOP",
        "Pointing_Up": "ATTENTION",
        "Victory": "PEACE",
    },
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9),
    (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

# ------------------------------------------------------------------ palette
# (light, dark) tuples - customtkinter switches them automatically.
BG = ("#F3F6F8", "#111318")
CARD = ("#FFFFFF", "#191D24")
CARD2 = ("#EDF1F4", "#222933")
FIELD = ("#F7F9FA", "#141A21")
GOLD = ("#087F79", "#38C9C1")
GOLD_H = ("#066A65", "#2BAFA8")
BURG = ("#176A73", "#246F78")
BURG_H = ("#10535B", "#2A848C")
TEXT = ("#18212B", "#E7EDF2")
SUB = ("#697586", "#A3ADB9")
ON_GOLD = ("#FFFFFF", "#0C1718")
ON_BURG = ("#FFFFFF", "#F3FAFA")
SEG_OFF = ("#E7ECEF", "#252D37")
SEG_OFF_H = ("#D8E0E5", "#303A46")
OK = ("#21865D", "#57C890")
WARN = ("#A96B11", "#E5B454")
BAD = ("#B54343", "#E27878")
ACTIVE = ("#E0F3F1", "#1D3A3B")


# ----------------------------------------------------------------- settings
def load_settings():
    cfg = copy.deepcopy(DEFAULTS)
    if SETTINGS_PATH.exists():
        try:
            user = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            for key, value in user.items():
                if key == "gesture_words" and isinstance(value, dict):
                    cfg[key].update({k: str(v) for k, v in value.items() if k in cfg[key]})
                elif key in cfg and not isinstance(cfg[key], dict):
                    cfg[key] = value
        except Exception as exc:
            print(f"[warn] settings.json unreadable ({exc}); using defaults")
    if cfg["mode"] not in MODES:
        cfg["mode"] = "Gestures"
    return cfg


def save_settings(cfg):
    try:
        SETTINGS_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[warn] could not save settings: {exc}")


# ------------------------------------------------------------------- models
def model_path(key):
    return MODEL_DIR / MODELS[key]["file"]


def model_ok(key):
    p = model_path(key)
    return p.exists() and p.stat().st_size >= MODELS[key]["min_size"]


def download_model(key, progress=None):
    info = MODELS[key]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = model_path(key).with_suffix(".part")
    req = urllib.request.Request(info["url"], headers={"User-Agent": "Mozilla/5.0 ArsSignorum"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if progress and total:
                progress(done / total)
    if tmp.stat().st_size < info["min_size"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("downloaded file is too small (blocked or wrong link)")
    tmp.replace(model_path(key))


def open_folder(path):
    path = str(path)
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as exc:
        print("could not open folder:", exc)


# ----------------------------------------------------------------- features
def hand_slots(hands):
    """[left_slot, right_slot]: each an (21,3) array or None."""
    slots = [None, None]
    if not hands:
        return slots
    if len(hands) == 1:
        h = hands[0]
        slots[1 if h["side"] == "Right" else 0] = h["pts"]
    else:
        ordered = sorted(hands[:2], key=lambda h: float(h["pts"][0, 0]))
        slots[0], slots[1] = ordered[0]["pts"], ordered[1]["pts"]
    return slots


def normalize_hand(pts):
    rel = pts - pts[0]
    scale = float(np.linalg.norm(rel[9]))
    return (rel / (scale if scale > 1e-6 else 1.0)).flatten()


def shape_features(slots):
    parts = [normalize_hand(p) if p is not None else np.zeros(63, np.float32) for p in slots]
    return np.concatenate(parts).astype(np.float32)


def position_features(slots):
    out = []
    for p in slots:
        out.extend([float(p[0, 0]), float(p[0, 1])] if p is not None else [0.0, 0.0])
    return np.array(out, dtype=np.float32)


def dynamic_features(frames, n):
    shapes = np.array([f[0] for f in frames], dtype=np.float32)
    poses = np.array([f[1] for f in frames], dtype=np.float32)
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    shapes, poses = shapes[idx], poses[idx]
    poses = poses - poses[0]
    return (np.concatenate([shapes.flatten(), poses.flatten() * 4.0]) / np.sqrt(n)).astype(np.float32)


def motion_amount(frames):
    """Measure both wrist travel and changes in finger/hand shape.

    Wrist-only movement rejected valid signs such as opening/closing a hand
    when the wrist stays almost stationary. Shape features capture those signs.
    """
    if len(frames) < 2:
        return 0.0

    poses = np.asarray([f[1] for f in frames], dtype=np.float32)
    wrist_motion = float((poses.max(axis=0) - poses.min(axis=0)).max())

    shapes = np.asarray([f[0] for f in frames], dtype=np.float32)
    if shapes.ndim != 2 or shapes.shape[0] < 2:
        return wrist_motion

    shape_change = np.mean(np.abs(shapes - shapes[0]), axis=1)
    finger_motion = float(np.max(shape_change)) * 3.0
    return max(wrist_motion, finger_motion)


def asl_features(pts, w, h):
    """Exactly the preprocessing the pretrained A-Z model was trained with."""
    xy = np.stack([np.minimum((pts[:, 0] * w).astype(int), w - 1),
                   np.minimum((pts[:, 1] * h).astype(int), h - 1)], axis=1).astype(np.float32)
    flat = (xy - xy[0]).flatten()
    peak = float(np.max(np.abs(flat)))
    if peak < 1e-6:
        return None
    return (flat / peak).astype(np.float32)


# --------------------------------------------------------------- classifiers
class KeypointClassifier:
    """Pretrained ASL A-Z classifier (TFLite). Works with any of three runtimes."""

    def __init__(self, path):
        interp = None
        for mod in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter", "tensorflow.lite"):
            try:
                interp = getattr(importlib.import_module(mod), "Interpreter")
                break
            except Exception:
                continue
        if interp is None:
            raise RuntimeError("no TFLite runtime installed (pip install ai-edge-litert)")
        content = Path(path).read_bytes()
        try:
            self.it = interp(model_content=content, num_threads=2)
        except TypeError:
            self.it = interp(model_content=content)
        self.it.allocate_tensors()
        self.input_detail = self.it.get_input_details()[0]
        self.output_detail = self.it.get_output_details()[0]
        self.i_idx = self.input_detail["index"]
        self.o_idx = self.output_detail["index"]
        self.input_shape = tuple(int(v) for v in self.input_detail["shape"])
        self.output_shape = tuple(int(v) for v in self.output_detail["shape"])
        self.input_size = int(np.prod(self.input_shape))
        self.output_size = int(np.prod(self.output_shape))

        if self.input_size != 42:
            raise RuntimeError(
                f"ASL model expects {self.input_size} input values; "
                "the application provides 42 hand-landmark values"
            )
        if self.output_size != len(ASL_LABELS):
            raise RuntimeError(
                f"ASL model returns {self.output_size} classes; expected {len(ASL_LABELS)} letters"
            )

    def predict(self, vec):
        values = np.asarray(vec, dtype=np.float32).reshape(-1)
        if values.size != self.input_size:
            raise ValueError(
                f"ASL feature size mismatch: expected {self.input_size}, got {values.size}"
            )

        input_dtype = self.input_detail["dtype"]
        if np.issubdtype(input_dtype, np.integer):
            scale, zero_point = self.input_detail.get("quantization", (0.0, 0))
            if scale:
                values = np.round(values / scale + zero_point)
        tensor = values.astype(input_dtype).reshape(self.input_shape)

        self.it.set_tensor(self.i_idx, tensor)
        self.it.invoke()
        out = np.squeeze(self.it.get_tensor(self.o_idx))

        if np.issubdtype(out.dtype, np.integer):
            scale, zero_point = self.output_detail.get("quantization", (0.0, 0))
            if scale:
                out = (out.astype(np.float32) - zero_point) * scale

        out = np.asarray(out, dtype=np.float32).reshape(-1)
        if out.size == 0 or not np.all(np.isfinite(out)):
            raise RuntimeError("ASL model returned an empty or invalid prediction")
        i = int(np.argmax(out))
        return i, float(out[i])


class SampleStore:
    def __init__(self, path, k):
        self.path, self.k = path, k
        self.X, self.y = None, np.array([], dtype="<U64")
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as z:
                    self.X, self.y = z["X"].astype(np.float32), z["y"].astype("<U64")
            except Exception as exc:
                print(f"[warn] could not load {path.name}: {exc}")

    def save(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if self.X is not None and len(self.y):
            temp_path = self.path.with_name(self.path.stem + ".tmp.npz")
            np.savez_compressed(temp_path, X=self.X, y=self.y)
            os.replace(temp_path, self.path)
        elif self.path.exists():
            self.path.unlink()

    def add(self, x, label):
        """Append a sample without replacing samples from existing labels."""
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        label = str(label).strip()
        if not label:
            raise ValueError("A training label cannot be empty")
        if self.X is None or len(self.y) == 0:
            self.X = x.copy()
            self.y = np.asarray([label], dtype="<U64")
            return
        if self.X.ndim != 2 or self.X.shape[1] != x.shape[1]:
            raise ValueError(
                f"Training feature size mismatch: saved {self.X.shape}, new {x.shape}"
            )
        self.X = np.concatenate((self.X, x), axis=0)
        self.y = np.concatenate((self.y, np.asarray([label], dtype="<U64")))

    def count(self, label=None):
        return int(len(self.y) if label is None else np.sum(self.y == label))

    def labels(self):
        return sorted(set(self.y.tolist()))

    def remove(self, label):
        keep = self.y != label
        self.X = self.X[keep] if self.X is not None else None
        self.y = self.y[keep]
        self.save()

    def clear(self):
        self.X, self.y = None, np.array([], dtype="<U64")
        self.save()

    def predict(self, x, max_dist):
        if self.X is None or len(self.y) == 0:
            return None, 0.0
        d = np.linalg.norm(self.X - x[None, :], axis=1)
        k = min(self.k, len(d))
        idx = np.argpartition(d, k - 1)[:k]
        idx = idx[d[idx] <= max_dist]
        if len(idx) == 0:
            return None, 0.0
        votes = {}
        for i in idx:
            votes[str(self.y[i])] = votes.get(str(self.y[i]), 0.0) + 1.0 / (float(d[i]) + 1e-3)
        label = max(votes, key=votes.get)
        return label, votes[label] / sum(votes.values())


class Stabilizer:
    """Turns noisy per-frame labels into one committed token per sign."""

    def __init__(self, hold, cooldown, repeat=0.0):
        self.hold, self.cooldown, self.repeat = max(1, int(hold)), cooldown, repeat
        self.current, self.count, self.gap = None, 0, 0
        self.last, self.last_t = None, 0.0

    def update(self, label, now):
        if label is None:
            self.gap += 1
            if self.gap >= 8:
                self.current, self.count, self.last = None, 0, None
            return None
        self.gap = 0
        if label == self.current:
            self.count += 1
        else:
            self.current, self.count = label, 1
        if self.count < self.hold or now - self.last_t < self.cooldown:
            return None
        if label != self.last or (self.repeat > 0 and now - self.last_t >= self.repeat):
            self.last, self.last_t = label, now
            return label
        return None

    def progress(self):
        return min(1.0, self.count / float(self.hold)) if self.current else 0.0


class Tracker:
    """MediaPipe Tasks wrapper: GestureRecognizer if available, else HandLandmarker."""

    def __init__(self, kind, path, cfg):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.mp, self.kind, self.last_ts = mp, kind, 0
        base = mp_python.BaseOptions(model_asset_buffer=Path(path).read_bytes())
        common = dict(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=float(cfg["min_detection"]),
            min_hand_presence_confidence=float(cfg["min_presence"]),
            min_tracking_confidence=float(cfg["min_tracking"]),
        )
        if kind == "gesture":
            self.impl = vision.GestureRecognizer.create_from_options(
                vision.GestureRecognizerOptions(**common))
        else:
            self.impl = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(**common))

    def process(self, rgb):
        ts = max(int(time.monotonic() * 1000), self.last_ts + 1)
        self.last_ts = ts
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.kind == "gesture":
            res = self.impl.recognize_for_video(img, ts)
        else:
            res = self.impl.detect_for_video(img, ts)
        hands = []
        for i, lm in enumerate(res.hand_landmarks or []):
            pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            side = None
            try:
                side = res.handedness[i][0].category_name
            except Exception:
                pass
            name, score = None, 0.0
            if self.kind == "gesture":
                try:
                    cat = res.gestures[i][0]
                    name, score = cat.category_name, float(cat.score)
                except Exception:
                    pass
            hands.append({"pts": pts, "side": side, "gesture": name, "gscore": score})
        return hands

    def close(self):
        try:
            self.impl.close()
        except Exception:
            pass


# ------------------------------------------------------------------- camera
class FakeCam:
    """Synthetic camera (SIGNLANG_FAKECAM=1) for testing the UI without hardware."""

    def __init__(self):
        self.t = 0

    def isOpened(self):
        return True

    def read(self):
        self.t += 1
        img = np.zeros((480, 640, 3), np.uint8)
        img[:] = (60 + int(40 * np.sin(self.t / 20)), 40, 50)
        cv2.putText(img, "SYNTHETIC CAMERA", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (220, 220, 220), 2)
        time.sleep(0.03)
        return True, img

    def set(self, *a):
        return True

    def release(self):
        pass


def _backends():
    return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]


def open_camera(index, resolution):
    if os.environ.get("SIGNLANG_FAKECAM"):
        return FakeCam()
    try:
        w, h = (int(v) for v in resolution.split("x"))
    except Exception:
        w, h = 1280, 720
    for backend in _backends():
        cap = cv2.VideoCapture(int(index), backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None


def probe_cameras(active, limit=6):
    if os.environ.get("SIGNLANG_FAKECAM"):
        return [0, 1]
    found = []
    for i in range(limit):
        if i == active:
            found.append(i)
            continue
        for backend in _backends()[:1]:
            cap = cv2.VideoCapture(i, backend)
            try:
                if cap.isOpened() and cap.read()[0]:
                    found.append(i)
            finally:
                cap.release()
    return found


# ------------------------------------------------------------------- engine
class Engine(threading.Thread):
    """Capture + recognition worker. The UI only reads snapshots and queued events."""

    def __init__(self, cfg, out_q):
        super().__init__(daemon=True, name="engine")
        self.cfg, self.q = cfg, out_q
        self.lock = threading.RLock()
        self.stop_flag = False
        self.cam, self.cam_dirty = None, True
        self.tracker, self.tracker_dirty = None, True
        self.tracker_error = self.asl_error = ""
        self.asl = None
        self.static = SampleStore(STATIC_FILE, int(cfg["knn_k"]))
        self.dynamic = SampleStore(DYNAMIC_FILE, int(cfg["knn_k"]))
        self.mode = cfg["mode"]
        self.frame, self.frame_id, self.info = None, 0, {"status": "Starting..."}
        self.stab = None
        # A shorter rolling window reduces motion-recognition latency.
        # Keep the feature resampling size unchanged so existing trained samples remain compatible.
        self.window = deque(maxlen=min(int(cfg["dynamic_window"]), 18))
        self.missing = 0
        self.dyn_candidate, self.dyn_block = None, 0.0
        self.rec = {"active": False}
        self.last_hand_t = time.time()
        self.space_sent = True
        self.fps, self.t_prev = 0.0, time.time()
        self.bad_reads, self.retry_at = 0, 0.0
        self.frame_no = 0
        self._new_stab()

    # ---- control API (called from UI thread)
    def stop(self):
        self.stop_flag = True

    def request_camera(self):
        self.cam_dirty = True

    def reload_models(self):
        self.tracker_dirty = True

    def set_mode(self, mode):
        with self.lock:
            self.mode = mode
            self.rec = {"active": False}
            self.window.clear()
            self.dyn_candidate = None
            self._new_stab()

    def apply_config(self):
        with self.lock:
            self.window = deque(self.window, maxlen=min(int(self.cfg["dynamic_window"]), 18))
            self.static.k = self.dynamic.k = int(self.cfg["knn_k"])
            self._new_stab()

    def start_hold(self, label):
        with self.lock:
            self.rec = {"active": True, "kind": "hold", "label": label, "n": 0}

    def stop_hold(self):
        with self.lock:
            if self.rec.get("active") and self.rec.get("kind") == "hold":
                label = self.rec["label"]
                self.rec = {"active": False}
                self.static.save()
                self.q.put(("samples",))
                self.q.put(("flash", f"Saved '{label}' ({self.static.count(label)} samples)"))

    def start_motion(self, label):
        with self.lock:
            start = time.time() + 1.5
            self.rec = {
                "active": True,
                "kind": "motion",
                "label": label,
                "frames": [],
                "start": start,
                "deadline": start + 15.0,
            }

    def _finish_motion_locked(self, timed_out=False):
        if not self.rec.get("active") or self.rec.get("kind") != "motion":
            return

        label = self.rec["label"]
        frames = list(self.rec.get("frames", []))
        self.rec = {"active": False}

        required = int(self.cfg["dynamic_resample"])
        if len(frames) < required:
            self.q.put(("flash", f"Not saved: detected {len(frames)} frames; at least {required} are needed."))
            self.q.put(("samples",))
            return

        movement = motion_amount(frames)
        if movement < float(self.cfg["min_motion"]):
            self.q.put(("flash", "Not saved: movement was too small. Repeat the sign with clearer motion."))
            self.q.put(("samples",))
            return

        self.dynamic.add(
            dynamic_features(frames, int(self.cfg["dynamic_resample"])),
            label
        )
        self.dynamic.save()
        self.q.put(("samples",))
        prefix = "Timed recording saved" if timed_out else "Saved motion"
        self.q.put(("flash", f"{prefix} '{label}' ({self.dynamic.count(label)} samples)"))

    def stop_motion(self):
        with self.lock:
            self._finish_motion_locked()

    def cancel_rec(self):
        with self.lock:
            self.rec = {"active": False}

    def delete_label(self, kind, label):
        with self.lock:
            (self.static if kind == "Hold" else self.dynamic).remove(label)
        self.q.put(("samples",))

    def clear_samples(self):
        with self.lock:
            self.static.clear()
            self.dynamic.clear()
        self.q.put(("samples",))

    def sample_summary(self):
        with self.lock:
            rows = [("Hold", lb, self.static.count(lb)) for lb in self.static.labels()]
            rows += [("Motion", lb, self.dynamic.count(lb)) for lb in self.dynamic.labels()]
        return rows

    def snapshot(self):
        with self.lock:
            return self.frame, self.frame_id, dict(self.info)

    def shutdown_save(self):
        with self.lock:
            self.static.save()
            self.dynamic.save()

    # ---- internals
    def _new_stab(self):
        c = self.cfg
        repeat = float(c["repeat_delay"]) if self.mode == "Alphabet" else 0.0
        self.stab = Stabilizer(int(c["hold_frames"]), float(c["cooldown"]), repeat)

    def _load_models(self):
        if self.tracker is not None:
            self.tracker.close()
        self.tracker, self.tracker_error, self.asl, self.asl_error = None, "", None, ""
        kind = "gesture" if model_ok("gesture") else ("landmark" if model_ok("hand") else None)
        if kind:
            try:
                self.tracker = Tracker(kind, model_path("gesture" if kind == "gesture" else "hand"), self.cfg)
            except Exception as exc:
                self.tracker_error = f"{type(exc).__name__}: {exc}"
        if model_ok("asl"):
            try:
                self.asl = KeypointClassifier(model_path("asl"))
            except Exception as exc:
                self.asl_error = str(exc)
        self.tracker_dirty = False

    def _open_cam(self):
        if self.cam is not None:
            self.cam.release()
            self.cam = None
        self.cam_dirty = False
        self.cam = open_camera(self.cfg["camera_index"], self.cfg["resolution"])
        self.bad_reads = 0
        if self.cam is None:
            self.retry_at = time.time() + 3
            self.q.put(("flash", f"Camera {self.cfg['camera_index']} could not be opened"))

    def _emit(self, token, kind, conf):
        self.q.put(("token", token, kind, conf, self.mode))

    def run(self):
        while not self.stop_flag:
            try:
                self._step()
            except Exception as exc:  # keep the worker alive
                with self.lock:
                    self.info = {"status": f"Engine error: {type(exc).__name__}: {exc}"}
                time.sleep(0.5)
        if self.cam is not None:
            self.cam.release()
        if self.tracker is not None:
            self.tracker.close()

    def _step(self):
        if self.tracker_dirty:
            self._load_models()
        if self.cam_dirty or (self.cam is None and time.time() >= self.retry_at):
            self._open_cam()
        if self.cam is None:
            with self.lock:
                self.frame = None
                self.info = {"status": "No camera - choose one in Settings", "hint": ""}
            time.sleep(0.2)
            return
        ok, frame = self.cam.read()
        if not ok:
            self.bad_reads += 1
            if self.bad_reads > 40:
                self.cam.release()
                self.cam, self.retry_at = None, time.time() + 2
                self.q.put(("flash", "Camera stopped delivering frames"))
            time.sleep(0.01)
            return
        self.bad_reads = 0
        cfg, now = self.cfg, time.time()
        self.frame_no += 1
        flipped = cv2.flip(frame, 1)  # the pretrained models expect a mirrored view
        h, w = flipped.shape[:2]
        hands = []
        if self.tracker is not None:
            hands = self.tracker.process(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))

        with self.lock:
            mode = self.mode
            pred, conf, source, hint, progress = None, 0.0, "", "", 0.0
            slots = hand_slots(hands)
            present = bool(hands)
            if present:
                self.last_hand_t, self.space_sent = now, False

            if self.tracker is None:
                hint = "No hand-tracking model - open the Models tab"
                if self.tracker_error:
                    hint += f" ({self.tracker_error})"
            elif mode == "Gestures" and self.tracker.kind != "gesture":
                hint = "Gesture Recognizer file missing - open the Models tab"
            elif mode == "Alphabet" and self.asl is None:
                hint = "ASL classifier unavailable: " + (self.asl_error or "model file missing (Models tab)")
            elif mode == "Custom Hold" and self.static.count() == 0 and not self.rec["active"]:
                hint = "No trained signs yet - open the Train tab"
            elif mode == "Custom Motion" and self.dynamic.count() == 0 and not self.rec["active"]:
                hint = "No trained motion signs yet - open the Train tab"

            if mode == "Gestures":
                best = None
                for hd in hands:
                    g = hd["gesture"]
                    if g and g not in ("None", "ILoveYou") and (best is None or hd["gscore"] > best["gscore"]):
                        best = hd
                if best and best["gscore"] >= float(cfg["gesture_min_score"]):
                    pred = cfg["gesture_words"].get(best["gesture"], best["gesture"].replace("_", " ").upper())
                    conf, source = best["gscore"], "gesture model"
            elif mode == "Alphabet":
                if hands and self.asl is not None:
                    vec = asl_features(hands[0]["pts"], w, h)
                    if vec is not None:
                        try:
                            idx, c = self.asl.predict(vec)
                            if idx >= len(ASL_LABELS):
                                hint = f"ASL model returned class {idx}; expected A-Z labels."
                            elif c >= float(cfg["alphabet_min_conf"]):
                                pred, conf, source = ASL_LABELS[idx], c, "ASL model"
                            else:
                                hint = f"Hand detected. Low ASL confidence ({c:.2f}); hold the sign clearly in view."
                        except Exception as exc:
                            self.asl_error = f"{type(exc).__name__}: {exc}"
                            hint = "ASL prediction failed: " + self.asl_error
            elif mode == "Custom Hold":
                shape = shape_features(slots)
                if present:
                    pred, conf = self.static.predict(shape, float(cfg["static_max_distance"]))
                    source = "trained"
                if self.rec["active"] and self.rec.get("kind") == "hold" and present and self.frame_no % 2 == 0:
                    lb = self.rec["label"]
                    self.static.add(shape, lb)
                    self.rec["n"] += 1
                    if self.rec["n"] % 25 == 0:
                        self.static.save()

            # --- commit logic
            rec_text = ""
            if self.rec["active"]:
                if self.rec.get("kind") == "hold":
                    rec_text = f"Recording '{self.rec['label']}' - {self.rec['n']} samples"
                elif self.rec.get("kind") == "motion":
                    if now < self.rec["start"]:
                        rec_text = f"Get ready... {self.rec['start'] - now:0.1f}s"
                    else:
                        if present:
                            self.rec["frames"].append((shape_features(slots), position_features(slots)))
                        n_need = max(
                            int(cfg["dynamic_window"]),
                            int(cfg["dynamic_resample"]) * 3
                        )
                        frame_count = len(self.rec["frames"])
                        rec_text = (
                            f"Recording '{self.rec['label']}' - {frame_count}/{n_need} frames. "
                            "Keep your hand visible and complete the movement."
                        )
                        if frame_count >= n_need:
                            self._finish_motion_locked()
                        elif now >= self.rec.get("deadline", now + 1):
                            self._finish_motion_locked(timed_out=True)
            elif mode == "Custom Motion":
                if present:
                    self.window.append((shape_features(slots), position_features(slots)))
                    self.missing = 0
                else:
                    self.missing += 1
                    if self.missing > 6:
                        self.window.clear()
                # Evaluate twice as often once the shorter motion window is full.
                if (len(self.window) == self.window.maxlen and self.frame_no % 2 == 0 and now >= self.dyn_block):
                    frames = list(self.window)
                    if motion_amount(frames) >= float(cfg["min_motion"]):
                        p, c = self.dynamic.predict(dynamic_features(frames, int(cfg["dynamic_resample"])),
                                                    float(cfg["dynamic_max_distance"]))
                        if p is not None and p == self.dyn_candidate:
                            self._emit(p, "word", c)
                            self.window.clear()
                            self.dyn_candidate, self.dyn_block = None, now + 1.5
                        else:
                            self.dyn_candidate = p
                        pred, conf, source = p, c, "trained"
                    else:
                        self.dyn_candidate = None
            else:
                token = self.stab.update(pred if not self.rec["active"] else None, now)
                progress = self.stab.progress()
                if token:
                    self._emit(token, "letter" if mode == "Alphabet" else "word", conf)

            # auto-space between letters when the hand leaves
            sp = float(cfg["auto_space_seconds"])
            if (mode == "Alphabet" and sp > 0 and not present and not self.space_sent
                    and now - self.last_hand_t >= sp):
                self.space_sent = True
                self.q.put(("space",))

            # --- draw overlay
            disp = flipped
            show_lm = bool(cfg["show_landmarks"])
            if show_lm:
                for hd in hands:
                    px = [(int(p[0] * w), int(p[1] * h)) for p in hd["pts"]]
                    for a, b in HAND_CONNECTIONS:
                        cv2.line(disp, px[a], px[b], (106, 168, 201), 2, cv2.LINE_AA)
                    for p in px:
                        cv2.circle(disp, p, 4, (43, 34, 142), -1, cv2.LINE_AA)
                        cv2.circle(disp, p, 4, (200, 220, 240), 1, cv2.LINE_AA)
            mirror = bool(cfg["mirror"])
            if not mirror:
                disp = cv2.flip(disp, 1)
            if pred and hands and cfg["show_overlay_text"]:
                xs = [p[0] * w for p in hands[0]["pts"]]
                ys = [p[1] * h for p in hands[0]["pts"]]
                x1, x2 = (min(xs), max(xs)) if mirror else (w - max(xs), w - min(xs))
                y1, y2 = min(ys), max(ys)
                cv2.rectangle(disp, (int(x1) - 12, int(y1) - 12), (int(x2) + 12, int(y2) + 12), (106, 168, 201), 2)
                tag = str(pred)
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
                tx, ty = int(x1) - 12, max(int(y1) - 20, th + 12)
                cv2.rectangle(disp, (tx, ty - th - 10), (tx + tw + 14, ty + 6), (43, 34, 142), -1)
                cv2.putText(disp, tag, (tx + 7, ty - 2), cv2.FONT_HERSHEY_DUPLEX, 0.9, (235, 240, 248), 2, cv2.LINE_AA)
            if w > 960:
                disp = cv2.resize(disp, (960, int(h * 960 / w)), interpolation=cv2.INTER_AREA)

            self.fps = 0.9 * self.fps + 0.1 / max(now - self.t_prev, 1e-6)
            self.t_prev = now
            self.frame = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            self.frame_id += 1
            self.info = {
                "status": "Live", "pred": pred, "conf": conf, "source": source, "hint": hint,
                "fps": self.fps, "hands": len(hands), "progress": progress, "rec_text": rec_text,
                "rec_active": self.rec["active"], "rec_kind": self.rec.get("kind"),
                "tracker": getattr(self.tracker, "kind", None), "mode": mode,
            }


# --------------------------------------------------------------------- TTS
def speak_async(text):
    def work():
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.say(text)
            eng.runAndWait()
        except Exception as exc:
            print("[tts]", exc)
    if text.strip():
        threading.Thread(target=work, daemon=True).start()


def apply_case(token, kind, current, style):
    if style == "UPPERCASE":
        return token.upper()
    if style == "lowercase":
        return token.lower()
    if style == "Capitalized":
        starts = (not current.strip()) or current.rstrip().endswith((".", "!", "?")) and current.endswith(" ")
        if kind == "letter":
            return token.upper() if starts else token.lower()
        return token.capitalize() if starts else token.lower()
    return token


# ------------------------------------------------------------------ UI parts
def gold_button(parent, text, command, width=100, **kw):
    return ctk.CTkButton(parent, text=text, command=command, width=width, fg_color=GOLD, hover_color=GOLD_H,
                         text_color=ON_GOLD, corner_radius=6, **kw)


def burg_button(parent, text, command, width=100, **kw):
    return ctk.CTkButton(parent, text=text, command=command, width=width, fg_color=BURG, hover_color=BURG_H,
                         text_color=ON_BURG, corner_radius=6, **kw)


def ghost_button(parent, text, command, width=100, **kw):
    return ctk.CTkButton(parent, text=text, command=command, width=width, fg_color="transparent",
                         hover_color=CARD2, text_color=GOLD, border_width=1, border_color=GOLD,
                         corner_radius=6, **kw)


def rule(parent):
    return ctk.CTkFrame(parent, height=1, fg_color=GOLD, corner_radius=0)


class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title(f"{APP_NAME} - Settings")
        self.geometry("560x720")
        self.minsize(520, 560)
        self.configure(fg_color=BG)
        self.transient(app)
        self.after(150, self.lift)
        F = app.fonts
        head = ctk.CTkFrame(self, fg_color=CARD, border_width=1, border_color=GOLD, corner_radius=0)
        head.pack(fill="x")
        ctk.CTkLabel(head, text="Settings", font=F["h1"], text_color=GOLD).pack(side="left", padx=18, pady=12)
        ghost_button(head, "Reset to defaults", self.reset, width=130).pack(side="right", padx=14)
        self.body = ctk.CTkScrollableFrame(self, fg_color=BG)
        self.body.pack(fill="both", expand=True, padx=10, pady=10)
        self._build()

    # ---- helpers
    def section(self, title):
        ctk.CTkLabel(self.body, text=title.upper(), font=self.app.fonts["small_b"], text_color=GOLD
                     ).pack(anchor="w", padx=8, pady=(16, 2))
        rule(self.body).pack(fill="x", padx=8, pady=(0, 6))

    def row(self, label):
        r = ctk.CTkFrame(self.body, fg_color="transparent")
        r.pack(fill="x", padx=8, pady=3)
        ctk.CTkLabel(r, text=label, text_color=TEXT, anchor="w").pack(side="left")
        return r

    def slider(self, label, key, lo, hi, steps, fmt="{:.2f}", cast=float, on_change=None):
        r = self.row(label)
        val = ctk.CTkLabel(r, text=fmt.format(self.app.cfg[key]), text_color=SUB, width=48)
        val.pack(side="right")

        def cb(v):
            v = cast(v)
            self.app.cfg[key] = v
            val.configure(text=fmt.format(v))
            self.app.cfg_changed(on_change)
        s = ctk.CTkSlider(r, from_=lo, to=hi, number_of_steps=steps, width=210, command=cb,
                          progress_color=GOLD, button_color=GOLD, button_hover_color=GOLD_H)
        s.set(self.app.cfg[key])
        s.pack(side="right", padx=8)

    def switch(self, label, key, on_change=None):
        r = self.row(label)
        var = ctk.BooleanVar(value=bool(self.app.cfg[key]))

        def cb():
            self.app.cfg[key] = bool(var.get())
            self.app.cfg_changed(on_change)
        ctk.CTkSwitch(r, text="", variable=var, command=cb, progress_color=GOLD, width=46).pack(side="right")

    def menu(self, label, key, values, on_change=None):
        r = self.row(label)

        def cb(v):
            self.app.cfg[key] = v
            self.app.cfg_changed(on_change)
        m = ctk.CTkOptionMenu(r, values=values, command=cb, width=170, fg_color=CARD2, button_color=GOLD,
                              button_hover_color=GOLD_H, text_color=TEXT, dropdown_fg_color=CARD,
                              dropdown_text_color=TEXT, dropdown_hover_color=CARD2)
        m.set(str(self.app.cfg[key]))
        m.pack(side="right")

    # ---- content
    def _build(self):
        for w in self.body.winfo_children():
            w.destroy()
        app = self.app
        self.section("Camera")
        r = self.row("Device")
        self.cam_menu = ctk.CTkOptionMenu(r, values=self._cam_values(), command=self._pick_cam, width=170,
                                          fg_color=CARD2, button_color=GOLD, button_hover_color=GOLD_H,
                                          text_color=TEXT, dropdown_fg_color=CARD, dropdown_text_color=TEXT,
                                          dropdown_hover_color=CARD2)
        self.cam_menu.set(f"Camera {app.cfg['camera_index']}")
        self.cam_menu.pack(side="right")
        ghost_button(r, "Scan", self._scan, width=60).pack(side="right", padx=8)
        self.menu("Resolution", "resolution", RESOLUTIONS, on_change="camera")
        self.switch("Mirror preview", "mirror")
        self.switch("Show hand skeleton", "show_landmarks")
        self.switch("Show sign label on video", "show_overlay_text")

        self.section("Recognition")
        self.slider("Hand detection confidence", "min_detection", 0.3, 0.95, 65, on_change="models")
        self.slider("Hand tracking confidence", "min_tracking", 0.3, 0.95, 65, on_change="models")
        self.slider("Gesture confidence needed", "gesture_min_score", 0.3, 0.95, 65)
        self.slider("Alphabet confidence needed", "alphabet_min_conf", 0.3, 0.99, 69)
        self.slider("Hold steady (frames)", "hold_frames", 3, 30, 27, fmt="{:.0f}", cast=int, on_change="stab")
        self.slider("Cooldown between words (s)", "cooldown", 0.2, 3.0, 28, fmt="{:.1f}", on_change="stab")
        self.slider("Repeat same letter after (s)", "repeat_delay", 0.8, 5.0, 42, fmt="{:.1f}", on_change="stab")
        self.slider("Auto-space when hand leaves (s, 0 = off)", "auto_space_seconds", 0.0, 5.0, 50, fmt="{:.1f}")

        self.section("Custom signs")
        self.slider("Hold-sign match strictness", "static_max_distance", 0.6, 3.0, 48)
        self.slider("Motion-sign match strictness", "dynamic_max_distance", 0.8, 4.0, 64)
        self.slider("Minimum hand movement", "min_motion", 0.03, 0.4, 37)

        self.section("Output")
        self.menu("Text style", "text_case", CASES)
        self.switch("Speak each new word aloud", "speak_on_word")

        self.section("Appearance")
        self.menu("Theme", "appearance", ["Dark", "Light", "System"], on_change="theme")

        self.section("Gesture vocabulary")
        ctk.CTkLabel(self.body, text="Choose the output text for each supported built-in gesture.",
                     text_color=SUB, wraplength=470, justify="left").pack(anchor="w", padx=8, pady=(0, 4))
        for gname in app.cfg["gesture_words"]:
            r = self.row(gname.replace("_", " "))
            e = ctk.CTkEntry(r, width=170, fg_color=FIELD, border_color=GOLD, text_color=TEXT)
            e.insert(0, app.cfg["gesture_words"][gname])
            e.pack(side="right")
            e.bind("<KeyRelease>", lambda _e, g=gname, ent=e: self._word(g, ent))

        self.section("Models")
        self.switch("Download missing models automatically", "auto_download")
        r = self.row("")
        ghost_button(r, "Open models folder", lambda: open_folder(MODEL_DIR), width=160).pack(side="right")

    def _word(self, gname, entry):
        self.app.cfg["gesture_words"][gname] = entry.get().strip() or gname
        self.app.cfg_changed()

    def _cam_values(self):
        vals = [f"Camera {i}" for i in self.app.cameras]
        cur = f"Camera {self.app.cfg['camera_index']}"
        if cur not in vals:
            vals.append(cur)
        return vals or ["Camera 0"]

    def _pick_cam(self, value):
        self.app.cfg["camera_index"] = int(value.split()[-1])
        self.app.cfg_changed("camera")

    def _scan(self):
        self.cam_menu.configure(values=["Scanning..."])
        self.app.scan_cameras(lambda: self.cam_menu.configure(values=self._cam_values()))

    def reset(self):
        keep = {k: self.app.cfg[k] for k in ("geometry",)}
        self.app.cfg.clear()
        self.app.cfg.update(copy.deepcopy(DEFAULTS))
        self.app.cfg.update(keep)
        self.app.cfg_changed("theme")
        self.app.cfg_changed("camera")
        self.app.cfg_changed("models")
        self._build()


class App(ctk.CTk):
    def __init__(self):
        self.cfg = load_settings()
        ctk.set_appearance_mode(self.cfg["appearance"])
        ctk.set_default_color_theme("dark-blue")
        super().__init__(fg_color=BG)
        self.title(f"{APP_NAME} - Sign Language Interpreter")
        try:
            self.geometry(self.cfg["geometry"])
        except Exception:
            self.geometry("1280x820")
        self.minsize(1120, 700)
        self.fonts = {
            "title": ctk.CTkFont(family="Segoe UI", size=25, weight="bold"),
            "h1": ctk.CTkFont(family="Segoe UI", size=21, weight="bold"),
            "h2": ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            "big": ctk.CTkFont(family="Segoe UI", size=48, weight="bold"),
            "serif": ctk.CTkFont(family="Segoe UI", size=16),
            "small": ctk.CTkFont(family="Segoe UI", size=12),
            "small_b": ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
        }
        self.q = queue.Queue()
        self.ui_q = queue.Queue()
        self.cameras = [self.cfg["camera_index"]]
        self.settings_win = None
        self.frame_seen = -1
        self.flash_until, self.hist_count = 0.0, 0
        self.dl_state = {}
        self.engine = Engine(self.cfg, self.q)

        self._build_header()
        self._build_body()
        self._build_status()
        self.bind("<Control-comma>", lambda e: self.open_settings())
        self.bind("<Control-t>", lambda e: self.toggle_theme())
        self.bind("<Control-s>", lambda e: self.save_text())
        self.bind("<F11>", lambda e: self.attributes("-fullscreen", not self.attributes("-fullscreen")))
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.engine.start()
        self.scan_cameras()
        self.refresh_models()
        self.refresh_samples()
        if self.cfg["auto_download"]:
            self.download_missing()
        self.after(60, self._tick)

    # ------------------------------------------------------------- layout

    def _card(self, parent, **kw):
        return ctk.CTkFrame(
            parent, fg_color=CARD, border_width=1,
            border_color=("#E2E8ED", "#2B333E"), corner_radius=12, **kw
        )

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=CARD, corner_radius=0, height=72)
        header.pack(fill="x")
        header.pack_propagate(False)

        brand = ctk.CTkFrame(header, fg_color="transparent")
        brand.pack(side="left", padx=22, pady=10)

        mark = ctk.CTkFrame(brand, width=42, height=42, corner_radius=12, fg_color=GOLD)
        mark.pack(side="left", padx=(0, 12))
        mark.pack_propagate(False)
        ctk.CTkLabel(
            mark, text="G", font=ctk.CTkFont(family="Segoe UI", size=23, weight="bold"),
            text_color=ON_GOLD
        ).place(relx=0.5, rely=0.5, anchor="center")

        names = ctk.CTkFrame(brand, fg_color="transparent")
        names.pack(side="left", anchor="center")
        ctk.CTkLabel(names, text="Gestify", font=self.fonts["title"], text_color=TEXT).pack(anchor="w")
        ctk.CTkLabel(
            names, text="ASL FINGERSPELLING  •  SIGN RECOGNITION",
            font=self.fonts["small_b"], text_color=SUB
        ).pack(anchor="w")

        actions = ctk.CTkFrame(header, fg_color="transparent")
        actions.pack(side="right", padx=20)
        ghost_button(actions, "Settings", self.open_settings, width=100).pack(side="right", padx=(8, 0))
        self.theme_btn = gold_button(actions, "Light theme", self.toggle_theme, width=112)
        self.theme_btn.pack(side="right")
        self._theme_label()
        ctk.CTkFrame(self, height=1, fg_color=("#E2E8ED", "#2B333E")).pack(fill="x")

    def _build_body(self):
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=16, pady=14)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        sidebar = self._card(body)
        sidebar.configure(corner_radius=14, width=218)
        sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 14))
        sidebar.grid_propagate(False)

        ctk.CTkLabel(sidebar, text="WORKSPACE", font=self.fonts["small_b"], text_color=SUB).pack(
            anchor="w", padx=16, pady=(20, 10)
        )

        self.nav_buttons = {}
        nav_items = [
            ("live", "◉", "Live recognition"),
            ("train", "✳", "Custom trainer"),
            ("history", "↶", "History"),
            ("models", "◇", "Model library"),
        ]
        for page, symbol, label in nav_items:
            button = ctk.CTkButton(
                sidebar, text=f"  {symbol}     {label}", anchor="w", height=42,
                corner_radius=9, fg_color="transparent", hover_color=CARD2,
                text_color=TEXT, font=ctk.CTkFont(family="Segoe UI", size=13),
                command=lambda p=page: self.show_page(p)
            )
            button.pack(fill="x", padx=9, pady=3)
            self.nav_buttons[page] = button

        ctk.CTkFrame(sidebar, height=1, fg_color=("#E2E8ED", "#2B333E")).pack(
            fill="x", padx=14, pady=(18, 12)
        )
        ctk.CTkLabel(sidebar, text="PREFERENCES", font=self.fonts["small_b"], text_color=SUB).pack(
            anchor="w", padx=16, pady=(0, 8)
        )
        ghost_button(sidebar, "Settings", self.open_settings, width=176, height=34).pack(
            fill="x", padx=12, pady=3
        )
        ctk.CTkLabel(
            sidebar, text="GESTIFY  •  DESKTOP",
            font=self.fonts["small_b"], text_color=SUB
        ).pack(side="bottom", anchor="w", padx=15, pady=15)

        main = ctk.CTkFrame(body, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        heading = ctk.CTkFrame(main, fg_color="transparent")
        heading.grid(row=0, column=0, sticky="ew", pady=(2, 12))
        self.page_title = ctk.CTkLabel(heading, text="Live recognition", font=self.fonts["h1"], text_color=TEXT)
        self.page_title.pack(anchor="w")
        self.page_subtitle = ctk.CTkLabel(
            heading, text="Recognize ASL signs in real time and build a transcript.",
            font=self.fonts["small"], text_color=SUB
        )
        self.page_subtitle.pack(anchor="w", pady=(2, 0))

        self.page_host = ctk.CTkFrame(main, fg_color="transparent")
        self.page_host.grid(row=1, column=0, sticky="nsew")
        self.page_host.grid_rowconfigure(0, weight=1)
        self.page_host.grid_columnconfigure(0, weight=1)
        self.pages = {}

        live = ctk.CTkFrame(self.page_host, fg_color="transparent")
        live.grid_rowconfigure(0, weight=1)
        live.grid_columnconfigure((0, 1), weight=1, uniform="live")
        self.pages["live"] = live

        camera_card = self._card(live)
        camera_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        camera_top = ctk.CTkFrame(camera_card, fg_color="transparent")
        camera_top.pack(fill="x", padx=16, pady=(14, 10))
        ctk.CTkLabel(camera_top, text="Camera preview", font=self.fonts["h2"], text_color=TEXT).pack(side="left")
        self.mode_btn = ctk.CTkSegmentedButton(
            camera_top, values=MODES, command=self.set_mode,
            selected_color=ACTIVE, selected_hover_color=ACTIVE,
            unselected_color=SEG_OFF, unselected_hover_color=SEG_OFF_H,
            text_color=TEXT, fg_color=SEG_OFF, corner_radius=8
        )
        self.mode_btn.set(self.cfg["mode"])
        self.mode_btn.pack(side="right")
        ctk.CTkFrame(camera_card, height=1, fg_color=("#E2E8ED", "#2B333E")).pack(fill="x", padx=15)
        self.video_holder = ctk.CTkFrame(
            camera_card, fg_color=("#E9EEF1", "#0D1117"), corner_radius=10
        )
        self.video_holder.pack(fill="both", expand=True, padx=14, pady=14)
        self.video_holder.pack_propagate(False)
        self.video_label = ctk.CTkLabel(
            self.video_holder, text="Waiting for camera...", text_color=SUB, font=self.fonts["serif"]
        )
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")
        self.hint_label = ctk.CTkLabel(
            camera_card, text="", text_color=WARN, wraplength=520, font=self.fonts["small"]
        )
        self.hint_label.pack(fill="x", padx=16, pady=(0, 12))

        translate_card = self._card(live)
        translate_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self._tab_translate(translate_card)
        self.conf_bar.configure(width=280)
        self.hold_bar.configure(width=280)

        train_page = ctk.CTkFrame(self.page_host, fg_color="transparent")
        train_page.grid_rowconfigure(0, weight=1)
        train_page.grid_columnconfigure((0, 1), weight=1, uniform="train")
        train_card = self._card(train_page)
        train_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._tab_train(train_card)

        train_camera_card = self._card(train_page)
        train_camera_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ctk.CTkLabel(
            train_camera_card, text="Live training camera",
            font=self.fonts["h2"], text_color=TEXT
        ).pack(anchor="w", padx=16, pady=(14, 4))
        ctk.CTkLabel(
            train_camera_card,
            text="Keep your hand visible in this preview while recording.",
            font=self.fonts["small"], text_color=SUB
        ).pack(anchor="w", padx=16, pady=(0, 10))
        self.train_video_holder = ctk.CTkFrame(
            train_camera_card, fg_color=("#E9EEF1", "#0D1117"), corner_radius=10
        )
        self.train_video_holder.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        self.train_video_holder.pack_propagate(False)
        self.train_video_label = ctk.CTkLabel(
            self.train_video_holder, text="Waiting for camera...",
            text_color=SUB, font=self.fonts["serif"]
        )
        self.train_video_label.place(relx=0.5, rely=0.5, anchor="center")
        self.train_rec_status = ctk.CTkLabel(
            train_camera_card, text="Camera ready when the live feed appears.",
            text_color=SUB, font=self.fonts["small"], wraplength=440
        )
        self.train_rec_status.pack(fill="x", padx=16, pady=(0, 14))
        self.pages["train"] = train_page

        history_page = ctk.CTkFrame(self.page_host, fg_color="transparent")
        history_page.grid_rowconfigure(0, weight=1)
        history_page.grid_columnconfigure(0, weight=1)
        history_card = self._card(history_page)
        history_card.grid(row=0, column=0, sticky="nsew")
        self._tab_history(history_card)
        self.pages["history"] = history_page

        models_page = ctk.CTkFrame(self.page_host, fg_color="transparent")
        models_page.grid_rowconfigure(0, weight=1)
        models_page.grid_columnconfigure(0, weight=1)
        models_card = self._card(models_page)
        models_card.grid(row=0, column=0, sticky="nsew")
        self._tab_models(models_card)
        self.pages["models"] = models_page

        self.show_page("live")

    def show_page(self, page):
        titles = {
            "live": ("Live recognition", "Recognize ASL signs in real time and build a transcript."),
            "train": ("Custom trainer", "Record and manage your own custom signs."),
            "history": ("Recognition history", "Review previous recognition events."),
            "models": ("Model library", "Manage the pretrained recognition models."),
        }
        if page not in self.pages:
            return
        for frame in self.pages.values():
            frame.grid_forget()
        self.pages[page].grid(row=0, column=0, sticky="nsew")
        title, subtitle = titles[page]
        self.page_title.configure(text=title)
        self.page_subtitle.configure(text=subtitle)
        for key, button in self.nav_buttons.items():
            active = key == page
            button.configure(
                fg_color=ACTIVE if active else "transparent",
                text_color=GOLD if active else TEXT,
                hover_color=CARD2
            )

    def _build_status(self):
        bar = ctk.CTkFrame(self, fg_color=CARD, corner_radius=0, height=32)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        ctk.CTkFrame(bar, height=1, fg_color=("#E2E8ED", "#2B333E")).pack(fill="x", side="top")
        self.status_left = ctk.CTkLabel(bar, text="", text_color=SUB, font=self.fonts["small"])
        self.status_left.pack(side="left", padx=16, pady=5)
        self.status_right = ctk.CTkLabel(bar, text="", text_color=GOLD, font=self.fonts["small"])
        self.status_right.pack(side="right", padx=16, pady=5)

    def _tab_translate(self, tab):
        ctk.CTkLabel(tab, text="DETECTED SIGN", font=self.fonts["small_b"], text_color=SUB).pack(pady=(10, 0))
        self.pred_label = ctk.CTkLabel(tab, text="-", font=self.fonts["big"], text_color=BURG)
        self.pred_label.pack(pady=(0, 2))
        self.src_label = ctk.CTkLabel(tab, text="", font=self.fonts["small"], text_color=SUB)
        self.src_label.pack()
        self.conf_bar = ctk.CTkProgressBar(tab, width=340, progress_color=GOLD, fg_color=CARD2)
        self.conf_bar.set(0)
        self.conf_bar.pack(pady=(8, 2))
        self.hold_bar = ctk.CTkProgressBar(tab, width=340, height=5, progress_color=BURG, fg_color=CARD2)
        self.hold_bar.set(0)
        self.hold_bar.pack(pady=(2, 0))
        ctk.CTkLabel(tab, text="confidence  /  hold steady to confirm", font=self.fonts["small"],
                     text_color=SUB).pack(pady=(2, 8))
        rule(tab).pack(fill="x", padx=14, pady=4)
        ctk.CTkLabel(tab, text="TRANSCRIPT", font=self.fonts["small_b"], text_color=SUB).pack(anchor="w", padx=16)
        self.text = ctk.CTkTextbox(tab, height=150, font=self.fonts["serif"], fg_color=FIELD, text_color=TEXT,
                                   border_width=1, border_color=GOLD, wrap="word")
        self.text.pack(fill="x", padx=14, pady=(2, 8))
        g = ctk.CTkFrame(tab, fg_color="transparent")
        g.pack(fill="x", padx=10)
        for c in range(3):
            g.grid_columnconfigure(c, weight=1)
        gold_button(g, "Space", lambda: self.insert_raw(" ")).grid(row=0, column=0, padx=4, pady=3, sticky="ew")
        burg_button(g, "Backspace", self.backspace).grid(row=0, column=1, padx=4, pady=3, sticky="ew")
        ghost_button(g, "Clear", self.clear_text).grid(row=0, column=2, padx=4, pady=3, sticky="ew")
        ghost_button(g, "Copy", self.copy_text).grid(row=1, column=0, padx=4, pady=3, sticky="ew")
        ghost_button(g, "Speak", self.speak_text).grid(row=1, column=1, padx=4, pady=3, sticky="ew")
        ghost_button(g, "Save...", self.save_text).grid(row=1, column=2, padx=4, pady=3, sticky="ew")

    def _tab_train(self, tab):
        ctk.CTkLabel(tab, text="Teach the app your own signs. Add a new label whenever you want to train another sign; existing labels are kept.",
                     text_color=SUB, wraplength=380, justify="left", font=self.fonts["small"]
                     ).pack(anchor="w", padx=14, pady=(10, 6))
        label_row = ctk.CTkFrame(tab, fg_color="transparent")
        label_row.pack(fill="x", padx=14, pady=4)
        self.label_entry = ctk.CTkEntry(
            label_row, placeholder_text="Sign label (e.g. THANK YOU)",
            fg_color=FIELD, border_color=GOLD, text_color=TEXT, height=34
        )
        self.label_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ghost_button(
            label_row, "New label", self.new_training_label, width=84, height=34
        ).pack(side="right")
        row = ctk.CTkFrame(tab, fg_color="transparent")
        row.pack(fill="x", padx=10)
        row.grid_columnconfigure((0, 1), weight=1)
        self.rec_hold_btn = gold_button(row, "Record hold sign", self.toggle_hold_record)
        self.rec_hold_btn.grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        self.rec_motion_btn = burg_button(row, "Record motion sign", self.record_motion)
        self.rec_motion_btn.grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        self.rec_label = ctk.CTkLabel(tab, text="Hold: show the sign, press record, move slightly, press stop.\n"
                                                "Motion: press record, wait for the countdown, perform it once.\n"
                                                "Record several examples for each label to improve recognition.",
                                      text_color=SUB, wraplength=380, justify="left", font=self.fonts["small"])
        self.rec_label.pack(anchor="w", padx=14, pady=6)
        rule(tab).pack(fill="x", padx=14, pady=4)
        head = ctk.CTkFrame(tab, fg_color="transparent")
        head.pack(fill="x", padx=14)
        ctk.CTkLabel(head, text="TRAINED SIGNS", font=self.fonts["small_b"], text_color=SUB).pack(side="left")
        ghost_button(head, "Delete all", self.clear_all_samples, width=80, height=24).pack(side="right")
        self.samples_frame = ctk.CTkScrollableFrame(tab, fg_color=FIELD, corner_radius=6, height=200)
        self.samples_frame.pack(fill="both", expand=True, padx=14, pady=(4, 10))

    def _tab_history(self, tab):
        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.pack(fill="x", padx=14, pady=(10, 4))
        ctk.CTkLabel(top, text="RECOGNITION LOG", font=self.fonts["small_b"], text_color=SUB).pack(side="left")
        ghost_button(top, "Clear", self.clear_history, width=70, height=24).pack(side="right")
        self.hist = ctk.CTkTextbox(tab, fg_color=FIELD, text_color=TEXT, border_width=1, border_color=GOLD,
                                   font=ctk.CTkFont(family="Consolas", size=12), wrap="none")
        self.hist.pack(fill="both", expand=True, padx=14, pady=(2, 12))
        self.hist.configure(state="disabled")

    def _tab_models(self, tab):
        ctk.CTkLabel(tab, text="Pretrained models live in the models folder. Download them here, or paste the "
                               "files in yourself.", text_color=SUB, wraplength=390, justify="left",
                     font=self.fonts["small"]).pack(anchor="w", padx=14, pady=(10, 6))
        self.model_rows = {}
        for key, info in MODELS.items():
            box = ctk.CTkFrame(tab, fg_color=CARD2, corner_radius=8)
            box.pack(fill="x", padx=12, pady=5)
            head = ctk.CTkFrame(box, fg_color="transparent")
            head.pack(fill="x", padx=10, pady=(8, 0))
            ctk.CTkLabel(head, text=info["title"], font=self.fonts["h2"], text_color=TEXT).pack(side="left")
            status = ctk.CTkLabel(head, text="", font=self.fonts["small_b"])
            status.pack(side="right")
            ctk.CTkLabel(box, text=info["purpose"], text_color=SUB, wraplength=380, justify="left",
                         font=self.fonts["small"]).pack(anchor="w", padx=10)
            ctk.CTkLabel(box, text="File: models/" + info["file"], text_color=SUB,
                         font=self.fonts["small"]).pack(anchor="w", padx=10)
            btns = ctk.CTkFrame(box, fg_color="transparent")
            btns.pack(fill="x", padx=6, pady=(4, 8))
            dl = gold_button(btns, "Download", lambda k=key: self.download_one(k), width=90, height=26)
            dl.pack(side="left", padx=4)
            ghost_button(btns, "Copy link", lambda k=key: self.copy_link(k), width=80, height=26).pack(side="left")
            self.model_rows[key] = (status, dl)
        bottom = ctk.CTkFrame(tab, fg_color="transparent")
        bottom.pack(fill="x", padx=12, pady=6)
        ghost_button(bottom, "Open models folder", lambda: open_folder(MODEL_DIR), width=150).pack(side="left", padx=4)
        ghost_button(bottom, "Re-check", self.recheck_models, width=80).pack(side="left", padx=4)
        ctk.CTkLabel(tab, text="Credits: MediaPipe (Google, Apache-2.0). ASL A-Z classifier by Muhib-Mehdi (MIT). "
                               "J and Z involve motion, so the static classifier is unreliable for them.",
                     text_color=SUB, wraplength=390, justify="left", font=self.fonts["small"]
                     ).pack(anchor="w", padx=14, pady=(6, 4))

    def _build_status(self):
        bar = ctk.CTkFrame(self, fg_color=CARD, border_width=1, border_color=GOLD, corner_radius=0, height=30)
        bar.pack(fill="x", side="bottom")
        self.status_left = ctk.CTkLabel(bar, text="", text_color=SUB, font=self.fonts["small"])
        self.status_left.pack(side="left", padx=14, pady=4)
        self.status_right = ctk.CTkLabel(bar, text="", text_color=GOLD, font=self.fonts["small"])
        self.status_right.pack(side="right", padx=14)

    # ------------------------------------------------------------ actions
    def flash(self, msg, secs=4.0):
        self.status_right.configure(text=msg)
        self.flash_until = time.time() + secs

    def _theme_label(self):
        dark = ctk.get_appearance_mode() == "Dark"
        self.theme_btn.configure(text="Light theme" if dark else "Dark theme")

    def toggle_theme(self):
        new = "Light" if ctk.get_appearance_mode() == "Dark" else "Dark"
        self.cfg["appearance"] = new
        ctk.set_appearance_mode(new)
        self._theme_label()
        save_settings(self.cfg)

    def open_settings(self):
        if self.settings_win is not None and self.settings_win.winfo_exists():
            self.settings_win.lift()
            self.settings_win.focus()
            return
        self.settings_win = SettingsWindow(self)

    def cfg_changed(self, what=None):
        if what == "camera":
            self.engine.request_camera()
        elif what == "models":
            self.engine.reload_models()
        elif what == "theme":
            ctk.set_appearance_mode(self.cfg["appearance"])
            self._theme_label()
        self.engine.apply_config()
        save_settings(self.cfg)

    def set_mode(self, mode):
        self.cfg["mode"] = mode
        self.mode_btn.set(mode)
        self.engine.set_mode(mode)
        self.rec_hold_btn.configure(text="Record hold sign")
        save_settings(self.cfg)

    def scan_cameras(self, done=None):
        def work():
            found = probe_cameras(int(self.cfg["camera_index"]))
            self.ui_q.put(("cameras", found, done))
        threading.Thread(target=work, daemon=True).start()

    # ---- transcript
    def get_text(self):
        return self.text.get("1.0", "end-1c")

    def insert_raw(self, s):
        self.text.insert("end", s)
        self.text.see("end")

    def append_token(self, token, kind):
        cur = self.get_text()
        token = apply_case(token, kind, cur, self.cfg["text_case"])
        if kind == "word":
            if cur and not cur.endswith((" ", "\n")):
                token = " " + token
            token += " "
        self.insert_raw(token)

    def backspace(self):
        if self.get_text():
            self.text.delete("end-2c", "end-1c")

    def clear_text(self):
        self.text.delete("1.0", "end")

    def copy_text(self):
        self.clipboard_clear()
        self.clipboard_append(self.get_text())
        self.flash("Copied to clipboard")

    def speak_text(self):
        speak_async(self.get_text())

    def save_text(self):
        text = self.get_text().strip()
        if not text:
            self.flash("Nothing to save yet")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="translation.txt",
                                            filetypes=[("Text file", "*.txt")])
        if path:
            Path(path).write_text(text + "\n", encoding="utf-8")
            self.flash(f"Saved {Path(path).name}")

    # ---- history
    def log(self, line):
        self.hist.configure(state="normal")
        self.hist.insert("end", line + "\n")
        self.hist.see("end")
        self.hist.configure(state="disabled")
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def clear_history(self):
        self.hist.configure(state="normal")
        self.hist.delete("1.0", "end")
        self.hist.configure(state="disabled")

    # ---- training
    def new_training_label(self):
        self.label_entry.delete(0, "end")
        self.label_entry.focus()
        self.flash("Enter a new label. Existing trained signs will be kept.")

    def _label(self):
        lb = self.label_entry.get().strip()
        if not lb:
            self.flash("Type a label first (Train tab)")
            self.label_entry.focus()
        return lb

    def toggle_hold_record(self):
        info = self.engine.snapshot()[2]
        if info.get("rec_active") and info.get("rec_kind") == "hold":
            self.engine.stop_hold()
            self.rec_hold_btn.configure(text="Record hold sign")
            return
        lb = self._label()
        if not lb:
            return
        if self.cfg["mode"] != "Custom Hold":
            self.set_mode("Custom Hold")
        self.engine.start_hold(lb)
        self.rec_hold_btn.configure(text="Stop recording")

    def record_motion(self):
        if self.rec_motion_btn.cget("text") == "Stop motion recording":
            self.engine.stop_motion()
            self.rec_motion_btn.configure(text="Record motion sign")
            return

        lb = self._label()
        if not lb:
            return
        if self.cfg["mode"] != "Custom Motion":
            self.set_mode("Custom Motion")
        self.engine.start_motion(lb)
        self.rec_motion_btn.configure(text="Stop motion recording")

    def refresh_samples(self):
        for w in self.samples_frame.winfo_children():
            w.destroy()
        rows = self.engine.sample_summary()
        if not rows:
            ctk.CTkLabel(self.samples_frame, text="Nothing trained yet.", text_color=SUB).pack(pady=14)
        for kind, label, n in rows:
            r = ctk.CTkFrame(self.samples_frame, fg_color="transparent")
            r.pack(fill="x", pady=2)
            ctk.CTkLabel(r, text=label, text_color=TEXT, font=self.fonts["h2"], anchor="w").pack(side="left", padx=6)
            ctk.CTkLabel(r, text=f"{kind} - {n}", text_color=SUB, font=self.fonts["small"]).pack(side="left", padx=4)
            burg_button(r, "Delete", lambda k=kind, lb=label: self.engine.delete_label(k, lb),
                        width=56, height=22).pack(side="right", padx=4)

    def clear_all_samples(self):
        self.engine.clear_samples()

    # ---- models
    def refresh_models(self):
        for key, (status, dl) in self.model_rows.items():
            if model_ok(key):
                status.configure(text="INSTALLED", text_color=OK)
                dl.configure(text="Re-download")
            else:
                state = self.dl_state.get(key)
                if state == "busy":
                    status.configure(text="DOWNLOADING...", text_color=WARN)
                elif state == "fail":
                    status.configure(text="DOWNLOAD FAILED - paste file manually", text_color=BAD)
                else:
                    status.configure(text="MISSING", text_color=BAD)
                dl.configure(text="Download")

    def recheck_models(self):
        self.dl_state.clear()
        self.refresh_models()
        self.engine.reload_models()
        self.flash("Models re-checked")

    def copy_link(self, key):
        self.clipboard_clear()
        self.clipboard_append(MODELS[key]["url"])
        self.flash(f"Link copied - save the file as models/{MODELS[key]['file']}")

    def download_one(self, key):
        if self.dl_state.get(key) == "busy":
            return
        self.dl_state[key] = "busy"
        self.refresh_models()

        def work():
            try:
                download_model(key)
                self.dl_state[key] = "ok"
                self.ui_q.put(("model_done", key, ""))
            except Exception as exc:
                self.dl_state[key] = "fail"
                self.ui_q.put(("model_done", key, str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def download_missing(self):
        need = [k for k in ("gesture", "asl") if not model_ok(k)]
        if need and not model_ok("gesture"):
            pass
        for k in need:
            self.download_one(k)

    # ---------------------------------------------------------------- loop
    def _tick(self):
        try:
            self._drain()
            frame, fid, info = self.engine.snapshot()
            self._show_frame(frame, fid, info)
            self._show_info(info)
        except Exception as exc:
            print("[ui]", type(exc).__name__, exc)
        if time.time() > self.flash_until and self.status_right.cget("text"):
            self.status_right.configure(text="")
        self.after(33, self._tick)

    def _drain(self):
        while True:
            try:
                ev = self.q.get_nowait()
            except queue.Empty:
                break
            kind = ev[0]
            if kind == "token":
                _, tok, tkind, conf, mode = ev
                self.append_token(tok, tkind)
                self.log(f"[{datetime.now():%H:%M:%S}] {mode:<13} {tok}  ({conf * 100:.0f}%)")
                if self.cfg["speak_on_word"] and tkind == "word":
                    speak_async(tok)
            elif kind == "space":
                cur = self.get_text()
                if cur and not cur.endswith((" ", "\n")):
                    self.insert_raw(" ")
            elif kind == "samples":
                self.refresh_samples()
                self.rec_hold_btn.configure(text="Record hold sign")
                self.rec_motion_btn.configure(text="Record motion sign")
            elif kind == "flash":
                self.flash(ev[1])
        while True:
            try:
                ev = self.ui_q.get_nowait()
            except queue.Empty:
                break
            if ev[0] == "cameras":
                self.cameras = ev[1] or [self.cfg["camera_index"]]
                if ev[2]:
                    ev[2]()
            elif ev[0] == "model_done":
                _, key, err = ev
                if err:
                    self.flash(f"Download failed: {err}", 8)
                else:
                    self.flash(f"{MODELS[key]['title']} installed")
                self.refresh_models()
                if model_ok(key):
                    self.engine.reload_models()

    def _render_preview(self, holder, label, frame):
        hw = holder.winfo_width() - 8
        hh = holder.winfo_height() - 8
        if hw < 50 or hh < 50:
            return
        fh, fw = frame.shape[:2]
        scale = min(hw / fw, hh / fh)
        size = (max(1, int(fw * scale)), max(1, int(fh * scale)))
        img = Image.fromarray(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
        photo = ctk.CTkImage(light_image=img, dark_image=img, size=size)
        label.configure(image=photo, text="")
        label._photo = photo

    def _show_frame(self, frame, fid, info):
        if frame is None:
            if self.frame_seen != -2:
                message = info.get("status", "Waiting for camera...")
                self.video_label.configure(image=None, text=message)
                self.train_video_label.configure(image=None, text=message)
                self.frame_seen = -2
            return
        if fid == self.frame_seen:
            return
        self.frame_seen = fid
        self._render_preview(self.video_holder, self.video_label, frame)
        self._render_preview(self.train_video_holder, self.train_video_label, frame)

    def _show_info(self, info):
        pred = info.get("pred")
        self.pred_label.configure(text=pred if pred else "-")
        self.conf_bar.set(max(0.0, min(1.0, info.get("conf", 0.0))))
        self.hold_bar.set(info.get("progress", 0.0))
        self.src_label.configure(text=info.get("source", "") or "")
        hint = info.get("hint", "")
        rec = info.get("rec_text", "")
        self.hint_label.configure(text=rec or hint, text_color=BAD if rec else WARN)
        self.train_rec_status.configure(
            text=rec or hint or "Camera ready. Choose a label and start recording.",
            text_color=BAD if rec else (WARN if hint else SUB)
        )
        if info.get("rec_active") is False:
            if self.rec_hold_btn.cget("text") == "Stop recording":
                self.rec_hold_btn.configure(text="Record hold sign")
            if self.rec_motion_btn.cget("text") == "Stop motion recording":
                self.rec_motion_btn.configure(text="Record motion sign")
        cam = self.cfg["camera_index"]
        tr = info.get("tracker") or "none"
        self.status_left.configure(
            text=f"{info.get('status', '')}   |   Camera {cam}   |   {info.get('fps', 0):.0f} FPS   |   "
                 f"hands: {info.get('hands', 0)}   |   tracker: {tr}   |   mode: {self.cfg['mode']}")

    def on_close(self):
        try:
            self.cfg["geometry"] = self.geometry().split("+")[0]
        except Exception:
            pass
        save_settings(self.cfg)
        self.engine.stop()
        self.engine.join(timeout=2)
        self.engine.shutdown_save()
        self.destroy()


def check():
    import tkinter
    print("Python", sys.version.split()[0], "|", platform.system())
    print("OpenCV", cv2.__version__, "| NumPy", np.__version__, "| customtkinter", ctk.__version__,
          "| Tk", tkinter.TkVersion)
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    assert hasattr(vision, "GestureRecognizer") and hasattr(vision, "HandLandmarker")
    print("MediaPipe", mp.__version__)
    for rt in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter", "tensorflow.lite"):
        try:
            importlib.import_module(rt)
            print("TFLite runtime:", rt)
            break
        except Exception:
            pass
    else:
        print("TFLite runtime: none (ASL Alphabet mode disabled; run: pip install ai-edge-litert)")
    print("Setup OK.")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        app = App()
        if "--selftest" in sys.argv:
            app.after(4000, app.on_close)
        app.mainloop()
