import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import time
from collections import deque

import numpy as np
import cv2
import hailo

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
BUFFER_SECONDS = 30          # How many seconds to buffer before trigger
FPS = 30                     # Pipeline framerate (adjust as needed)
MAX_BUFFER = BUFFER_SECONDS * FPS
VIDEO_DIR = "gravacoes"      # Directory to save clips
VIDEO_PREFIX = "braco_cruzado_"

os.makedirs(VIDEO_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Helper: COCO keypoint indices
# ----------------------------------------------------------------------------
def get_keypoints():
    return {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16,
    }

# ----------------------------------------------------------------------------
# User callback: maintains buffer and saving logic
# ----------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.frame_buffer = deque(maxlen=MAX_BUFFER)
        self.crossed = False
        self.last_save_time = 0

    def add_frame(self, frame):
        # Store a copy to avoid mutation
        self.frame_buffer.append(frame.copy())

    def save_buffered_clip(self, width, height):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(VIDEO_DIR, f"{VIDEO_PREFIX}{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filename, fourcc, FPS, (width, height))

        # Annotate last frame
        if self.frame_buffer:
            last_frame = self.frame_buffer[-1]
            cv2.putText(last_frame,
                        "ARMS CROSSED",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (0, 255, 0),
                        4,
                        cv2.LINE_AA)
            self.frame_buffer[-1] = last_frame

        for f in self.frame_buffer:
            writer.write(f)
        writer.release()

        print(f"[INFO] Saved clip: {filename}")
        self.last_save_time = time.time()
        self.crossed = True

# ----------------------------------------------------------------------------
# Main GStreamer callback
# ----------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Frame counting
    user_data.increment()

    # Get format & raw frame
    fmt, width, height = get_caps_from_pad(pad)
    frame = None
    if user_data.use_frame and fmt and width and height:
        rgb_frame = get_numpy_from_buffer(buffer, fmt, width, height)
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        user_data.add_frame(frame)

    # Parse detections and keypoints
    roi = hailo.get_roi_from_buffer(buffer)
    dets = roi.get_objects_typed(hailo.HAILO_DETECTION)
    kp_map = get_keypoints()

    crossed_detected = False
    for det in dets:
        if det.get_label() != "person":
            continue
        bbox = det.get_bbox()
        landmarks = det.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks:
            continue
        pts = landmarks[0].get_points()

        # Compute wrist & nose positions
        lw = pts[kp_map['left_wrist']]
        rw = pts[kp_map['right_wrist']]
        nose = pts[kp_map['nose']]
        lx = int((lw.x()*bbox.width() + bbox.xmin()) * width)
        ly = int((lw.y()*bbox.height()+bbox.ymin()) * height)
        rx = int((rw.x()*bbox.width() + bbox.xmin()) * width)
        ry = int((rw.y()*bbox.height()+bbox.ymin()) * height)
        nx = int((nose.x()*bbox.width()+bbox.xmin()) * width)
        ny = int((nose.y()*bbox.height()+bbox.ymin()) * height)

        # Draw wrist indicators
        if frame is not None:
            cv2.circle(frame, (lx, ly), 5, (0, 255, 0), -1)
            cv2.circle(frame, (rx, ry), 5, (0, 255, 0), -1)

        # Check: both wrists above nose & left wrist to the right
        if ly < ny and ry < ny and lx > rx:
            crossed_detected = True
            break

    # Save buffer if newly crossed and sufficient time passed
    now = time.time()
    if crossed_detected and not user_data.crossed and (now - user_data.last_save_time) > BUFFER_SECONDS:
        user_data.save_buffered_clip(width, height)

    # Overlay status text
    if user_data.use_frame and frame is not None:
        status = "ARMS CROSSED" if crossed_detected else "ARMS NOT CROSSED"
        cv2.putText(frame,
                    status,
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 255, 0),
                    4,
                    cv2.LINE_AA)
        user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    Gst.init(None)
    user_data = user_app_callback_class()
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()
