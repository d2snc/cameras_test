#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pose-estimation with arms-crossed detection and asynchronous
hardware-accelerated video recorder for Raspberry Pi.
LED gives short blinks while the file is being saved.
2025-06-13
"""

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import cv2, numpy as np, hailo
import os, time, threading, multiprocessing as mp
from collections import deque
from datetime import datetime

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# ──────────────────────────────────────────────────────────
#  GPIO 17 setup (on-board LED, BCM numbering)
# ──────────────────────────────────────────────────────────
try:
    import gpiod
    _chip = gpiod.Chip("gpiochip0")
    _led  = _chip.get_line(17)
    _led.request(consumer="pose-led", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
    LED_OK = True
    print("GPIO17 LED ready")
except Exception as e:
    LED_OK = False
    print("GPIO17 LED disabled:", e)

def pulse_led(duration=0.15):
    """Turn LED on for ‹duration› s (non-blocking)."""
    if not LED_OK:
        return
    _led.set_value(1)
    threading.Timer(duration, lambda: _led.set_value(0), daemon=True).start()

# ──────────────────────────────────────────────────────────
#  LED blinker used during “saving video…”
# ──────────────────────────────────────────────────────────
def _blink_led_forever(stop_event, on_ms=150, period_ms=750):
    """Blink continuously until stop_event.set() is called."""
    while not stop_event.is_set():
        pulse_led(on_ms / 1000.0)
        stop_event.wait(period_ms / 1000.0)

# ──────────────────────────────────────────────────────────
#  Asynchronous saver process
# ──────────────────────────────────────────────────────────
SAVE_DIR = "gravacoes"
os.makedirs(SAVE_DIR, exist_ok=True)

def saver_proc(frame_q: mp.Queue, stop_q: mp.Queue, width, height, fps):
    """
    Receives JPEG-encoded frames on frame_q, writes a hardware-encoded
    H.264 MP4. Terminates after receiving a None frame or when stop_q
    gets a signal and the queue is empty.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(SAVE_DIR, f"arms_crossed_{ts}.mp4")
    print(f"[Saver] Writing {outfile}")

    gst_cmd = (
        f"gst-launch-1.0 -q appsrc name=appsrc is-live=true block=true "
        f"format=3 do-timestamp=true ! jpegdec ! "
        f"videoconvert ! video/x-raw,format=I420,width={width},height={height},"
        f"framerate={fps}/1 ! "
        f"v4l2h264enc extra-controls=\"encode,frame_level_rate_control_enable=true\" "
        f"keyframe-period={fps*2} ! h264parse ! mp4mux ! filesink location={outfile}"
    )
    try:
        import subprocess, shlex
        gst_proc = subprocess.Popen(shlex.split(gst_cmd), stdin=subprocess.PIPE)
        use_gst = True
    except Exception as e:
        use_gst = False
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(outfile, fourcc, fps, (width, height))
        print("[Saver] HW encoder unavailable, falling back to CPU:", e)

    frames = 0
    while True:
        # exit if main process signalled and queue is empty
        if stop_q.poll() and frame_q.empty():
            break

        try:
            jpeg = frame_q.get(timeout=0.1)
        except Exception:
            continue
        if jpeg is None:
            break

        if use_gst:
            gst_proc.stdin.write(jpeg)
        else:
            img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            vw.write(img)

        frames += 1

    if use_gst:
        gst_proc.stdin.close()
        gst_proc.wait()
    else:
        vw.release()
    print(f"[Saver] Done – {frames} frames saved")

# ──────────────────────────────────────────────────────────
#  Detection utilities
# ──────────────────────────────────────────────────────────
KP = {"nose": 0, "left_wrist": 9, "right_wrist": 10}

def arms_crossed(points, bbox, W, H):
    """
    True if both wrists are above nose and left-wrist x > right-wrist x.
    """
    try:
        lw = points[KP["left_wrist"]]
        rw = points[KP["right_wrist"]]
        nz = points[KP["nose"]]

        lwx = (lw.x()*bbox.width() + bbox.xmin()) * W
        lwy = (lw.y()*bbox.height() + bbox.ymin()) * H
        rwx = (rw.x()*bbox.width() + bbox.xmin()) * W
        rwy = (rw.y()*bbox.height() + bbox.ymin()) * H
        nzy = (nz.y()*bbox.height() + bbox.ymin()) * H

        return (lwy < nzy and rwy < nzy) and (lwx > rwx)
    except Exception:
        return False

# ──────────────────────────────────────────────────────────
#  Callback class
# ──────────────────────────────────────────────────────────
class UserCB(app_callback_class):
    FPS               = 30
    BUFFER_FRAMES     = FPS * 40        # 40 s pre-event
    AFTER_FRAMES      = FPS * 5         # 5 s post-event
    COOLDOWN_SEC      = 5

    def __init__(self):
        super().__init__()
        self.buf  = deque(maxlen=self.BUFFER_FRAMES)
        self.trig = False
        self.after = 0
        self.last_evt = 0.0
        self.total = 0
        self.dim = None                  # (h,w)
        # saver-related
        self.saver = None
        self.frame_q = self.stop_q = None
        self.blink_stop = None
        self.status = "Arms Not Crossed"

    # ──────────────── saver helpers ────────────────
    def _start_saver(self, h, w):
        self.frame_q = mp.Queue(maxsize=500)
        self.stop_q  = mp.Queue()
        self.saver   = mp.Process(target=saver_proc,
                                  args=(self.frame_q, self.stop_q, w, h, self.FPS),
                                  daemon=True)
        self.saver.start()
        # LED blink thread
        self.blink_stop = threading.Event()
        threading.Thread(target=_blink_led_forever,
                         args=(self.blink_stop,), daemon=True).start()

    def _send_frame(self, frame):
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok and self.saver and self.saver.is_alive():
            try:
                self.frame_q.put_nowait(jpg.tobytes())
            except mp.queues.Full:
                pass  # drop if queue is full

# ──────────────────────────────────────────────────────────
#  GStreamer pad probe
# ──────────────────────────────────────────────────────────
def app_cb(pad, info, user: UserCB):
    buf = info.get_buffer()
    fmt, W, H = get_caps_from_pad(pad)
    if buf is None or fmt is None:
        return Gst.PadProbeReturn.OK

    frame = get_numpy_from_buffer(buf, fmt, W, H)

    # Detect arms crossed
    roi = hailo.get_roi_from_buffer(buf)
    crossed_now = False
    for det in roi.get_objects_typed(hailo.HAILO_DETECTION):
        if det.get_label() != "person":
            continue
        lms = det.get_objects_typed(hailo.HAILO_LANDMARKS)
        if lms and arms_crossed(lms[0].get_points(), det.get_bbox(), W, H):
            crossed_now = True
            break

    t = time.time()
    if crossed_now:
        user.status = "Arms Crossed!"
        if not user.trig and t - user.last_evt > UserCB.COOLDOWN_SEC:
            user.trig = True
            user.after = 0
            user.last_evt = t
            user.total += 1
            pulse_led(0.3)
            print("*** Arms crossed – capturing ***")
    else:
        user.status = "Arms Not Crossed"

    # Handle post-trigger frame counting
    if user.trig:
        user.after += 1
        if user.after >= UserCB.AFTER_FRAMES:
            user.trig = False
            user.stop_q.put(True)        # tell saver to finalise
            if user.blink_stop:          # stop LED blinking
                user.blink_stop.set()
            print("*** capture complete – writing file ***")

    # ──────────────── drawing & buffering ────────────────
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, user.status, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (0,0,255) if "Crossed" in user.status else (0,255,0), 2)
    cv2.putText(bgr, f"Detections: {user.total}", (12, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

    # keep circular buffer
    user.buf.append(bgr)

    # launch saver on first frame after trigger
    if user.trig and user.after == 1 and user.saver is None:
        h, w = bgr.shape[:2]
        user.dim = (h, w)
        user._start_saver(h, w)
        # dump pre-event history
        for f in list(user.buf)[-UserCB.BUFFER_FRAMES:]:
            user._send_frame(f)
    # during saving, keep feeding frames
    elif user.saver and user.saver.is_alive():
        user._send_frame(bgr)

    user.set_frame(bgr)
    return Gst.PadProbeReturn.OK

# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    Gst.init(None)
    print("=== Pose Detection – Arms Crossed ===")
    print(f"Saving to: {os.path.abspath(SAVE_DIR)}")
    print("Ctrl-C to quit.\n")

    try:
        user = UserCB()
        app  = GStreamerPoseEstimationApp(app_cb, user)
        app.run()
    finally:
        if LED_OK:
            _led.set_value(0)
