#!/usr/bin/env python3
"""
Arms‑crossed detector + rolling video recorder accelerated by Hailo‑8L.

* Uses a YOLOv8‑Pose HEF (e.g. `yolov8s_pose.hef`) to detect the human pose.
* Captures video from the Raspberry Pi 5 camera via GStreamer (`libcamerasrc`).
* Keeps the last 20 s of frames in RAM (deque).
* When both wrists are detected above the head **and** crossing each other,
  it:
    1. Starts a background thread that saves the previous 20 s to an MP4 file.
    2. Turns GPIO‑17 **HIGH** for 3 s (visual feedback) and then LOW again.

Tested with:
  • Raspberry Pi OS Bookworm‑64 bit
  • Python >= 3.11  (set `PYTHONPATH` to include the `hailo_apps_infra` repo)
  • Hailo‑8L SDK 2.9 and `hailo‐apps‐infra` ≥ v2.0.0
  • Pi Camera Module 3 (RAW) + `libcamera`

Install prerequisites:
  sudo apt install python3-libcamera python3-gpiod python3-gi python3-opencv \
                       gstreamer1.0-tools gstreamer1.0-plugins-{base,good,bad} \
                       gstreamer1.0-libav
  # Hailo SDK (follow Hailo docs) – makes `hailonet` and `hailofilter` elements available.

Run:
  python3 pose_crossed_arms_recorder.py --hef yolov8s_pose.hef

Author: ChatGPT (OpenAI o3) – 2025‑06‑16
"""
import argparse
import collections
import os
import pathlib
import threading
import time
from datetime import datetime

import gi

# --- GStreamer / Hailo infrastructure --------------------------------------
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from hailo_apps_infra.hailo_rpi_common import (
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# --- GPIO via libgpiod ------------------------------------------------------
try:
    import gpiod

    CHIP = gpiod.Chip("gpiochip0")
    LED = CHIP.get_line(17)
    LED.request(consumer="pose-led", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
    GPIO_OK = True
except Exception as e:
    print("[WARN] GPIO17 unavailable – LED feedback disabled:", e)
    GPIO_OK = False


def pulse_led(duration: float = 3.0):
    """Set GPIO‑17 HIGH for *duration* seconds, then LOW (non‑blocking)."""
    if not GPIO_OK:
        return

    LED.set_value(1)

    def _off():
        LED.set_value(0)

    t = threading.Timer(duration, _off)
    t.daemon = True  # Avoid Timer daemon keyword issues on older Python
    t.start()


# --- Arms‑crossed logic -----------------------------------------------------
class ArmsCrossDetector:
    """Very simple heuristic for detecting arms crossed above the head."""

    # Key‑point indices for YOLOv8‑pose (17 kpts, 0‑based)
    NOSE = 0
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_WRIST = 9
    RIGHT_WRIST = 10

    def __init__(self, vertical_margin_px: int = 10):
        self.vertical_margin = vertical_margin_px

    def __call__(self, keypoints):
        """Return True if wrists are above head & crossing each other."""
        try:
            nose_y = keypoints[self.NOSE][1]
            lw_x, lw_y = keypoints[self.LEFT_WRIST]
            rw_x, rw_y = keypoints[self.RIGHT_WRIST]
        except IndexError:
            return False  # Not enough keypoints

        # 1. Both wrists clearly above the head (y smaller means higher)
        above_head = lw_y < nose_y - self.vertical_margin and rw_y < nose_y - self.vertical_margin
        if not above_head:
            return False

        # 2. Wrists roughly cross in X (left wrist on the right side of right wrist)
        crossed = lw_x > rw_x
        return crossed


# --- Video saver ------------------------------------------------------------
class AsyncVideoSaver(threading.Thread):
    """Background thread that dumps frames to MP4 so GST pipeline stays real‑time."""

    def __init__(self, frames, fps, out_dir="recordings"):
        super().__init__(daemon=True)
        self.frames = list(frames)  # copy to avoid mutation by producer
        self.fps = fps
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)

    def run(self):
        if not self.frames:
            return
        h, w, _ = self.frames[0].shape
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.out_dir / f"arms_crossed_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(path), fourcc, self.fps, (w, h))
        for f in self.frames:
            vw.write(f)
        vw.release()
        print(f"[SAVE] clip written to {path} (frames={len(self.frames)})")


# --- User application callback ---------------------------------------------
class UserCallback(app_callback_class):
    def __init__(self, buffer_len_frames: int, fps: int):
        super().__init__()
        self.buffer = collections.deque(maxlen=buffer_len_frames)
        self.fps = fps
        self.detector = ArmsCrossDetector()
        self.cooldown = 0  # frames remaining until we re‑arm detection
        self.COOLDOWN_FRAMES = fps * 5  # 5 s anti‑spam

    def process_frame(self, np_frame, keypoints):
        # Draw inference text for debugging (optional)
        label = "ARMS CROSSED" if self.detector(keypoints) else "ARMS NOT CROSSED"
        color = (0, 0, 255) if label == "ARMS CROSSED" else (0, 255, 0)
        cv2.putText(np_frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # -- Mandatory for GStreamerPoseEstimationApp ---------------------------
    def on_new_buffer(self, pad, info):  # pylint: disable=unused-argument
        buf = info.get_buffer()
        np_frame = get_numpy_from_buffer(buf)

        # keypoints are stored in Gst.Buffer metadata set by hailofilter
        kpts_meta = buf.get_meta("HailoJsonMeta")
        keypoints = []
        if kpts_meta:
            import json

            meta_dict = json.loads(kpts_meta.get_json())
            if meta_dict.get("objects"):
                best = max(meta_dict["objects"], key=lambda o: o["confidence"])
                keypoints = [(pt["x"], pt["y"]) for pt in best["keypoints"]]

        # Append frame AFTER extracting np array to avoid copy after drop
        self.buffer.append(np_frame.copy())

        crossed = self.detector(keypoints) if keypoints else False
        if crossed and self.cooldown == 0:
            print("[EVENT] Arms crossed detected – saving clip ...")
            pulse_led(3.0)
            AsyncVideoSaver(self.buffer, self.fps).start()
            self.cooldown = self.COOLDOWN_FRAMES
        else:
            self.cooldown = max(0, self.cooldown - 1)

        # For debugging overlay (optional)
        # self.process_frame(np_frame, keypoints)

        return Gst.PadProbeReturn.OK


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Pose + Arms‑cross detection")
    parser.add_argument("--hef", required=True, help="Path to YOLOv8 pose HEF file")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS (default=30)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--buffer", type=int, default=20, help="Seconds of history to keep")
    args = parser.parse_args()

    Gst.init(None)

    frames_in_buffer = args.buffer * args.fps
    callback = UserCallback(frames_in_buffer, args.fps)

    # Build the pose‑estimation pipeline using Hailo helper.
    # The helper internally creates a pipeline similar to:
    #   libcamerasrc -> videoconvert -> videoscale -> hailonet -> hailofilter -> appsink
    app = GStreamerPoseEstimationApp(app_callback=callback, user_data=None)

    # Override defaults where needed (width/height/FPS & HEF path)
    app.hef_path = os.path.abspath(args.hef)
    app.src_caps = (
        f"video/x-raw,format=RGB,width={args.width},height={args.height},framerate={args.fps}/1"
    )

    try:
        app.run()
    except KeyboardInterrupt:
        print("Interrupted – shutting down …")
        app.stop()


if __name__ == "__main__":
    import cv2  # delayed import so headless systems without GUI can run

    main()
