"""Shared helpers for the capture and checker apps (camera + cropping + speech)."""
import subprocess
import sys

import cv2

# Fraction of the frame's shorter side used as the square "apple area" in the middle.
# Must be the same when you capture training photos and when you check apples.
DEFAULT_ROI = 0.8


def open_camera(index=0, width=1280, height=720):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {index}. Is it plugged in? Try --camera 1 or --camera 2.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(5):  # let auto-exposure settle
        cap.read()
    return cap


def center_square(frame, roi=DEFAULT_ROI):
    """Return the square crop in the middle of the frame and its box (x, y, side)."""
    h, w = frame.shape[:2]
    side = int(min(h, w) * roi)
    x0, y0 = (w - side) // 2, (h - side) // 2
    return frame[y0:y0 + side, x0:x0 + side], (x0, y0, side)


def draw_box(frame, box, color=(255, 255, 255)):
    x, y, s = box
    cv2.rectangle(frame, (x, y), (x + s, y + s), color, 2)


def put_text(img, text, org, scale=0.8, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def speak(text):
    """Say text aloud without blocking. Uses the voice built into Windows, so nothing to install."""
    safe = text.replace("'", "")
    try:
        if sys.platform.startswith("win"):
            ps = ("Add-Type -AssemblyName System.Speech; "
                  f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')")
            subprocess.Popen(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps],
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        elif sys.platform == "darwin":
            subprocess.Popen(["say", safe])
        else:
            subprocess.Popen(["espeak", safe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, OSError):
        pass  # no speech engine: the on-screen result still works
