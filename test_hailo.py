#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pose-Estimation + “arms-crossed” trigger for Hailo-8 on Raspberry Pi 5  
──────────────────────────────────────────────────────────────────────
• Listens to GStreamerPoseEstimationApp (hailo_apps_infra).  
• Keeps a rolling 22-second frame-buffer (≈20 s before + 2 s depois).  
• When the operator crosses both wrists above the head for ≥0.8 s:
      – pulses GPIO-17 LED for 3 s (libgpiod)  
      – saves the buffered clip to gravacoes/AAAAMMDD_HHMMSS.avi  
• Optional on-screen OSD showing “ARMS CROSSED / ARMS NOT CROSSED”.  
Tested under Bullseye 64-bit with Python 3.11, GStreamer 1.22 and
hailo-rt-sdk 4.26.  Requires:
    sudo apt install python3-gpiod python3-opencv python3-gi
"""
# ───── Imports ──────────────────────────────────────────────────────
import gi, os, cv2, time, threading
from collections import deque
from datetime import datetime

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import numpy as np
import hailo
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad, get_numpy_from_buffer, app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# ───── GPIO via libgpiod ────────────────────────────────────────────
try:
    import gpiod
    CHIP_NAME = "gpiochip0"
    LED_LINE_OFFSET = 17
    _chip = gpiod.Chip(CHIP_NAME)
    _led  = _chip.get_line(LED_LINE_OFFSET)
    _led.request(consumer="pose-led", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
    LED_OK = True
except Exception as e:
    print("GPIO17 LED disabled:", e)
    LED_OK = False

_last_led_pulse = 0.0
def pulse_led(duration: float = 3.0):
    """Light GPIO-17 for *duration* seconds (non-blocking)."""
    global _last_led_pulse
    if not LED_OK: return
    t_now = time.time()
    if t_now - _last_led_pulse < 0.2:  # debounce
        return
    _last_led_pulse = t_now
    _led.set_value(1)
    threading.Timer(duration, lambda: _led.set_value(0), daemon=True).start()

# ───── Helper: COCO keypoints map ───────────────────────────────────
def get_keypoints():
    return {
        'nose': 0,
        'left_eye': 1,   'right_eye': 2,
        'left_ear': 3,   'right_ear': 4,
        'left_shoulder': 5,  'right_shoulder': 6,
        'left_elbow': 7,     'right_elbow': 8,
        'left_wrist': 9,     'right_wrist': 10,
        'left_hip': 11,      'right_hip': 12,
        'left_knee': 13,     'right_knee': 14,
        'left_ankle': 15,    'right_ankle': 16,
    }
KP = get_keypoints()

# ───── Parameters ──────────────────────────────────────────────────
BUFFER_SECONDS = 22            # 20 s antes + 2 s depois
FPS_FALLBACK   = 30            # usado até FPS real ser medido
DETECTION_HOLD = 0.8           # seg. que os braços devem ficar cruzados
POST_SECONDS   = 2.0           # quanto tempo gravar depois do gatilho
SAVE_DIR       = "gravacoes"
os.makedirs(SAVE_DIR, exist_ok=True)

# ───── Frame buffer + state container ──────────────────────────────
class PoseAppState(app_callback_class):
    """Extends hailo app_callback_class with circular buffer & logic."""
    def __init__(self):
        super().__init__()
        self.deque_frames   = deque(maxlen=FPS_FALLBACK*BUFFER_SECONDS)
        self.deque_time     = deque(maxlen=FPS_FALLBACK*BUFFER_SECONDS)
        self.fps_estimate   = FPS_FALLBACK
        self.last_fps_calc  = time.time()
        self.frame_counter  = 0

        # trigger/detection
        self.cross_start_ts = None
        self.triggered      = False
        self.save_running   = False
        self.save_until_ts  = 0.0

    # ───────── Frame push ──────────────────────────────────────────
    def push_frame(self, frame):
        now = time.time()
        self.deque_frames.append(frame.copy())
        self.deque_time.append(now)
        # dynamic FPS update every second
        self.frame_counter += 1
        if now - self.last_fps_calc >= 1.0:
            self.fps_estimate = self.frame_counter / (now - self.last_fps_calc)
            target_len = int(self.fps_estimate * BUFFER_SECONDS)
            if target_len != self.deque_frames.maxlen:
                self.deque_frames = deque(self.deque_frames, maxlen=target_len)
                self.deque_time   = deque(self.deque_time,   maxlen=target_len)
            self.frame_counter = 0
            self.last_fps_calc = now

    # ───────── Persist clip ───────────────────────────────────────
    def save_clip_async(self):
        if self.save_running: return
        self.save_running = True

        frames_to_write = list(self.deque_frames)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname  = os.path.join(SAVE_DIR, f"bracos_{ts_str}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        h, w   = frames_to_write[0].shape[:2]
        writer = cv2.VideoWriter(fname, fourcc,
                                 max(10, min(self.fps_estimate, 30)),
                                 (w, h))
        if not writer.isOpened():
            print("‼️  VideoWriter failed")
            self.save_running = False
            return
        print(f"💾 Salvando vídeo {fname} ({len(frames_to_write)} frames)…")
        for f in frames_to_write:
            writer.write(f)
        writer.release()
        print(f"✅ Vídeo salvo: {fname}")
        self.save_running = False

# ───── Pose / arms-crossed heuristics ──────────────────────────────
def arms_crossed(kp_abs, kp_conf, frame_w):
    """
    kp_abs  – dict {name:(x,y)} in *pixels* (image coordinates)
    kp_conf – dict {name:confidence_bool}
    """
    needed = ['left_wrist','right_wrist','left_shoulder','right_shoulder','nose']
    if not all(kp_conf.get(k, False) for k in needed):
        return False

    lw, rw = kp_abs['left_wrist'], kp_abs['right_wrist']
    ls, rs = kp_abs['left_shoulder'], kp_abs['right_shoulder']
    nose   = kp_abs['nose']

    wrists_above_head = lw[1] < nose[1] and rw[1] < nose[1]
    wrists_cross      = (lw[0] > rs[0] and rw[0] < ls[0] and lw[0] > rw[0])
    close_horiz       = abs(lw[0]-rw[0]) < frame_w * 0.30

    return wrists_above_head and wrists_cross and close_horiz

# ───── GST Pad-probe callback ──────────────────────────────────────
def gst_callback(pad: Gst.Pad, info: Gst.PadProbeInfo, state: PoseAppState):
    buf = info.get_buffer()
    if buf is None:
        return Gst.PadProbeReturn.OK

    fmt, W, H = get_caps_from_pad(pad)
    if fmt is None or W is None or H is None:
        return Gst.PadProbeReturn.OK

    # Convert frame to numpy (RGB)
    frame = get_numpy_from_buffer(buf, fmt, W, H)
    kp_map = {}
    kp_conf = {}

    # Parse Hailo detections/landmarks
    roi = hailo.get_roi_from_buffer(buf)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    crossed_now = False
    for det in detections:
        if det.get_label() != "person":  # ignore non-person
            continue
        landmarks = det.get_objects_typed(hailo.HAILO_LANDMARKS)
        if len(landmarks) == 0:
            continue
        pts = landmarks[0].get_points()
        bbox = det.get_bbox()
        # Build absolute coordinates dict
        for name, idx in KP.items():
            if idx >= len(pts): continue
            p = pts[idx]
            x = int((p.x() * bbox.width()  + bbox.xmin()) * W)
            y = int((p.y() * bbox.height() + bbox.ymin()) * H)
            kp_map[name]   = (x, y)
            kp_conf[name]  = p.score() > 0.5 if hasattr(p, "score") else True

        crossed_now = arms_crossed(kp_map, kp_conf, W) or crossed_now

    # ── OSD (optional)──────────────────────────────────────────────
    if state.use_frame:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        text = "ARMS CROSSED" if crossed_now else "ARMS NOT CROSSED"
        color = (0,0,255) if crossed_now else (0,255,0)
        cv2.putText(frame_bgr, text, (30,60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, color, 3, cv2.LINE_AA)
        state.set_frame(frame_bgr)
        frame_to_buffer = frame_bgr
    else:
        frame_to_buffer = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # ── Buffer & trigger state machine ─────────────────────────────
    now = time.time()
    state.push_frame(frame_to_buffer)

    if crossed_now:
        if state.cross_start_ts is None:
            state.cross_start_ts = now
        elif (now - state.cross_start_ts >= DETECTION_HOLD) and not state.triggered:
            # Trigger!
            state.triggered = True
            state.save_until_ts = now + POST_SECONDS
            pulse_led(3.0)
            print("⚠️  Braços cruzados detectados – iniciando contagem para salvar vídeo")
    else:
        state.cross_start_ts = None

    # When post-window expires, dump clip
    if state.triggered and now >= state.save_until_ts:
        state.triggered = False
        threading.Thread(target=state.save_clip_async, daemon=True).start()

    return Gst.PadProbeReturn.OK

# ───── Entrypoint ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("▶️  Inicializando pipeline Hailo-8… (Ctrl-C para sair)")
    app_state = PoseAppState()
    # We want frames for buffer & OSD
    app_state.use_frame = True
    gst_app = GStreamerPoseEstimationApp(gst_callback, app_state)
    try:
        gst_app.run()
    except KeyboardInterrupt:
        print("\n⏹️  Encerrando…")
