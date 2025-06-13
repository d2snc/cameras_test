#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pose estimation + arms-crossed detector, Raspberry Pi
2025-06-13  rev.5   (30-second clip limit)
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import cv2, numpy as np, hailo, os, time, threading, multiprocessing as mp
import subprocess, shlex, signal
from collections import deque
from datetime import datetime

# ---------------------------------------------------------------------------------
# GPIO (BCM-17 LED) ----------------------------------------------------------------
# ---------------------------------------------------------------------------------
try:
    import gpiod
    _chip = gpiod.Chip("gpiochip0")
    _led  = _chip.get_line(17)
    _led.request(consumer="pose-led", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
    LED_OK = True
except Exception as e:
    print("GPIO17 LED disabled:", e)
    LED_OK = False

def pulse_led(duration=3.0):
    if not LED_OK: return
    _led.set_value(1)
    t = threading.Timer(duration, lambda: _led.set_value(0))
    t.daemon = True
    t.start()

# ---------------------------------------------------------------------------------
# Saver process (unchanged) --------------------------------------------------------
# ---------------------------------------------------------------------------------
def saver_proc(frame_q: mp.Queue, stop_q: mp.Queue, w, h, fps):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mp4_path = os.path.join("gravacoes", f"arms_crossed_{ts}.mp4")
    print(f"[Saver] Starting file {mp4_path}")

    def start_hw():
        gst_cmd = (
            f"gst-launch-1.0 -q appsrc name=appsrc is-live=true block=true format=3 do-timestamp=true ! "
            f"jpegdec ! videoconvert ! video/x-raw,format=I420,width={w},height={h},framerate={fps}/1 ! "
            f"v4l2h264enc extra-controls=\"encode,frame_level_rate_control_enable=true\" keyframe-period={fps*2} ! "
            f"h264parse ! mp4mux ! filesink location={mp4_path}"
        )
        try:
            return subprocess.Popen(shlex.split(gst_cmd), stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    preexec_fn=os.setsid)
        except FileNotFoundError:
            return None

    proc = start_hw()
    use_gst = proc is not None
    if not use_gst:
        print("[Saver] HW encoder unavailable → OpenCV fallback.")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))

    frames = 0
    while True:
        if not stop_q.empty():
            frame_q.put(None)
        try:
            jpeg_bytes = frame_q.get(timeout=0.05)
        except Exception:
            jpeg_bytes = None
        if jpeg_bytes is None:
            break

        if use_gst:
            try: proc.stdin.write(jpeg_bytes)
            except (BrokenPipeError, OSError):
                print("[Saver] HW pipeline died → OpenCV fallback.")
                try: os.killpg(proc.pid, signal.SIGINT)
                except Exception: pass
                use_gst = False
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vw = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
        if not use_gst:
            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
            vw.write(frame)
        frames += 1

    if use_gst:
        proc.stdin.close(); proc.wait(timeout=5)
    else:
        vw.release()
    print(f"[Saver] Saved {frames} frames → {mp4_path}")

# ---------------------------------------------------------------------------------
# Hailo helpers --------------------------------------------------------------------
# ---------------------------------------------------------------------------------
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad, get_numpy_from_buffer, app_callback_class
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

KP = {"nose": 0, "left_wrist": 9, "right_wrist": 10}
def crossed(points, bbox, W, H):
    try:
        lw, rw, nz = points[9], points[10], points[0]
        lwx, lwy = (lw.x()*bbox.width()+bbox.xmin())*W, (lw.y()*bbox.height()+bbox.ymin())*H
        rwx, rwy = (rw.x()*bbox.width()+bbox.xmin())*W, (rw.y()*bbox.height()+bbox.ymin())*H
        nzy      = (nz.y()*bbox.height()+bbox.ymin())*H
        return (lwy < nzy and rwy < nzy) and (lwx > rwx)
    except Exception:
        return False

def dump_history(buf, feed_fn):
    for f in buf: feed_fn(f, block=True)

# ---------------------------------------------------------------------------------
# User callback --------------------------------------------------------------------
# ---------------------------------------------------------------------------------
class UserCB(app_callback_class):
    FPS               = 30
    BUFFER            = 600      ### CHANGED → 20 s pre-event (600 /30 fps)
    AFTER_TRIGGER     = 300      ### CHANGED → 10 s post-event (300 /30 fps)
    COOLDOWN_S        = 5

    def __init__(self):
        super().__init__()
        self.buf, self.dim = deque(maxlen=self.BUFFER), None
        self.last_event, self.triggered, self.after_count = 0.0, False, 0
        self.saver, self.frame_q, self.stop_q = None, None, None
        self.status, self.tot = "Arms Not Crossed", 0

    def _launch_saver(self, h, w):
        self.frame_q, self.stop_q = mp.Queue(maxsize=800), mp.Queue()
        self.saver = mp.Process(target=saver_proc,
                                args=(self.frame_q, self.stop_q, w, h, self.FPS),
                                daemon=True)
        self.saver.start()

    def _feed(self, frame, *, block=False):
        if not (self.saver and self.saver.is_alive()): return
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok: return
        try: self.frame_q.put(jpg.tobytes(), block=block, timeout=0.02)
        except mp.queues.Full: pass

# ---------------------------------------------------------------------------------
def app_cb(pad, info, user: UserCB):
    buf = info.get_buffer(); fmt,W,H = get_caps_from_pad(pad)
    if buf is None or fmt is None: return Gst.PadProbeReturn.OK
    frame = get_numpy_from_buffer(buf, fmt, W, H)

    roi  = hailo.get_roi_from_buffer(buf)
    dets = roi.get_objects_typed(hailo.HAILO_DETECTION)
    now_crossed = any(
        d.get_label()=="person" and d.get_objects_typed(hailo.HAILO_LANDMARKS) and
        crossed(d.get_objects_typed(hailo.HAILO_LANDMARKS)[0].get_points(), d.get_bbox(), W, H)
        for d in dets)

    t = time.time()
    if now_crossed:
        user.status = "Arms Crossed!"
        if not user.triggered and (t - user.last_event) > UserCB.COOLDOWN_S:
            user.triggered = True; user.after_count = 0
            user.last_event = t; user.tot += 1
            pulse_led(3); print("*** Arms crossed – capture started ***")
    else: user.status = "Arms Not Crossed"

    if user.triggered:
        user.after_count += 1
        if user.after_count >= UserCB.AFTER_TRIGGER:
            user.triggered = False
            if user.stop_q: user.stop_q.put(True)
            if LED_OK: _led.set_value(0)
            print("*** Capture finished – writing file ***")

    # overlay
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, user.status, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,0,255) if "Crossed" in user.status else (0,255,0), 2)
    cv2.putText(bgr, f"Detections: {user.tot}", (10,75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

    user.buf.append(bgr)
    if user.triggered and user.after_count == 1:
        if user.dim is None: user.dim = bgr.shape[:2]
        h,w = user.dim; user._launch_saver(h,w)
        threading.Thread(target=dump_history,
                         args=(list(user.buf), user._feed),
                         daemon=True).start()
    elif user.saver and user.saver.is_alive():
        user._feed(bgr)

    user.set_frame(bgr)
    return Gst.PadProbeReturn.OK

# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    Gst.init(None); os.makedirs("gravacoes", exist_ok=True)
    try:
        user = UserCB()
        app  = GStreamerPoseEstimationApp(app_cb, user)
        print("Running – press Ctrl-C to quit."); app.run()
    finally:
        if LED_OK: _led.set_value(0)
