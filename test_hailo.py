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
        'left_wrist': 9,
        'right_wrist': 10,
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

    def save_buffered_clip(self, width, height):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(VIDEO_DIR, f"{VIDEO_PREFIX}{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filename, fourcc, FPS, (width, height))

        for frame in self.frame_buffer:
            writer.write(frame)
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

    user_data.increment()

    # Get raw frame
    fmt, width, height = get_caps_from_pad(pad)
    frame = None
    if user_data.use_frame and fmt and width and height:
        rgb = get_numpy_from_buffer(buffer, fmt, width, height)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Detect crossed arms
    crossed = False
    if frame is not None:
        roi = hailo.get_roi_from_buffer(buffer)
        dets = roi.get_objects_typed(hailo.HAILO_DETECTION)
        kp_map = get_keypoints()
        for det in dets:
            if det.get_label() != "person":
                continue
            bbox = det.get_bbox()
            landmarks = det.get_objects_typed(hailo.HAILO_LANDMARKS)
            if not landmarks:
                continue
            pts = landmarks[0].get_points()
            lw = pts[kp_map['left_wrist']]
            rw = pts[kp_map['right_wrist']]
            nose = pts[kp_map['nose']]
            lx = int((lw.x()*bbox.width() + bbox.xmin()) * width)
            ly = int((lw.y()*bbox.height()+bbox.ymin()) * height)
            rx = int((rw.x()*bbox.width() + bbox.xmin()) * width)
            ry = int((rw.y()*bbox.height()+bbox.ymin()) * height)
            nx = int((nose.x()*bbox.width()+bbox.xmin()) * width)
            ny = int((nose.y()*bbox.height()+bbox.ymin()) * height)
            if ly < ny and ry < ny and lx > rx:
                crossed = True
                break

        # Overlay status on frame
        status = "ARMS CROSSED" if crossed else "ARMS NOT CROSSED"
        cv2.putText(frame,
                    status,
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 255, 0),
                    4,
                    cv2.LINE_AA)
        user_data.set_frame(frame)
        self_frame = frame.copy()
        # Add annotated frame to buffer
        user_data.frame_buffer.append(self_frame)

    # Save buffer if crossed
    now = time.time()
    if crossed and not user_data.crossed and (now - user_data.last_save_time) > BUFFER_SECONDS:
        user_data.save_buffered_clip(width, height)

    return Gst.PadProbeReturn.OK

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    Gst.init(None)
    user_data = user_app_callback_class()
    user_data.use_frame = True
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()
