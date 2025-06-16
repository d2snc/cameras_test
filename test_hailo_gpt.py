#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hailo Pose-Estimation – braços cruzados
► buffer circular sem cópias • gravação MJPG • pipeline na main-thread
Jun 2025  rev. 2b
"""

import gi, os, cv2, hailo, time, threading, numpy as np, psutil, gc
from collections import deque
from datetime import datetime
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

# ───────────────────────── HAILO APPS INFRA ────────────────────────────────
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# ───────────────────────── CONFIGURAÇÃO ────────────────────────────────────
BUFFER_SECONDS       = 20
DETECTION_INTERVAL   = 0.25          # ~4 Hz
POSE_HOLD_SECONDS    = 0.8
OUTPUT_DIR           = "recordings"
FILE_PREFIX_AUTO     = "arms_"
FILE_PREFIX_MANUAL   = "manual_"
MAX_RECORD_FPS       = 20
BUFFER_SCALE         = 0.4
CODEC_FOURCC         = cv2.VideoWriter_fourcc(*"MJPG")

# ───────────────────────── LED opcional ────────────────────────────────────
LED_AVAILABLE = False
try:
    import gpiod
    CHIP, LINE = "gpiochip4", 17
    _chip = gpiod.Chip(CHIP)
    _led  = _chip.get_line(LINE)
    _led.request(consumer="hailo-led",
                 type=gpiod.LINE_REQ_DIR_OUT,
                 default_vals=[0])
    LED_AVAILABLE = True
except Exception as e:
    print("[LED] desativado:", e)

def pulse_led(sec=3.0):
    if LED_AVAILABLE:
        _led.set_value(1)
        threading.Timer(sec, lambda: _led.set_value(0), daemon=True).start()

# ────────────────────── Fallback RGB extraction ────────────────────────────
def gstbuffer_to_rgb(buf, w, h):
    ok, info = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None
    try:
        need = w * h * 3
        if len(info.data) < need:
            return None
        arr = np.frombuffer(info.data[:need], dtype=np.uint8).reshape((h, w, 3))
        return arr.copy()
    finally:
        buf.unmap(info)

# ───────────────────────── Estado global ───────────────────────────────────
class State(app_callback_class):
    def __init__(self):
        super().__init__()
        self.buf_sec   = BUFFER_SECONDS
        self.fps_est   = 30.0
        self.frames    = deque(maxlen=int(self.fps_est * self.buf_sec))
        self.stamps    = deque(maxlen=int(self.fps_est * self.buf_sec))
        self.lock      = threading.Lock()
        # detecção & gravação
        self.pose_start = None
        self.pose_ready = False
        self.recording  = False
        self.trigger_fr = None
        self.auto_cnt   = 0
        self.manual_cnt = 0

S = State()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ───────────────────── Thread de gravação ──────────────────────────────────
def writer_thread(frames, fname, fps):
    try:
        os.nice(-10)
    except:
        pass
    if not frames:
        S.recording = False
        return
    h, w = frames[0].shape[:2]
    if BUFFER_SCALE < 1.0:
        w, h = int(w / BUFFER_SCALE), int(h / BUFFER_SCALE)
    fps = max(10.0, min(fps, MAX_RECORD_FPS))
    vw  = cv2.VideoWriter(fname, CODEC_FOURCC, fps, (w, h))
    if not vw.isOpened():
        print("[ERR] VideoWriter não iniciou")
        S.recording = False
        return
    for i, f in enumerate(frames):
        if BUFFER_SCALE < 1.0:
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        if i and i % max(1, len(frames) // 10) == 0:
            print(f"[SAVE] {i / len(frames) * 100:.0f}%…")
    vw.release()
    print("[SAVE] OK:", fname)
    S.recording = False
    gc.collect()

def save_buffer(manual=False):
    if S.recording:
        return
    with S.lock:
        if not S.frames:
            return
        frames_for_write = list(S.frames)   # referências, sem cópias
        S.frames.clear();  S.stamps.clear() # libera RAM
    tag   = FILE_PREFIX_MANUAL if manual else FILE_PREFIX_AUTO
    fname = os.path.join(OUTPUT_DIR,
                         f"{tag}{datetime.now():%Y%m%d_%H%M%S}.avi")
    S.recording = True
    threading.Thread(target=writer_thread,
                     args=(frames_for_write, fname, S.fps_est),
                     daemon=True).start()
    pulse_led()

# ───────────────────── Callback GStreamer ─────────────────────────────────
def app_callback(pad, info, _):
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK
    now = time.time()
    S.increment()

    # FPS médio + resize
    if not hasattr(app_callback, "_t0"):
        app_callback._t0, app_callback._cnt = now, 0
    app_callback._cnt += 1
    if now - app_callback._t0 >= 1.0:
        S.fps_est = app_callback._cnt / (now - app_callback._t0)
        app_callback._cnt, app_callback._t0 = 0, now
        new_len = int(S.fps_est * S.buf_sec)
        if new_len != S.frames.maxlen:
            with S.lock:
                S.frames = deque(S.frames, maxlen=new_len)
                S.stamps = deque(S.stamps, maxlen=new_len)
            print(f"[BUF] → {new_len} quadros ({S.buf_sec}s @ {S.fps_est:.1f} fps)")

    # extrai frame se necessário
    need_frame = (BUFFER_SCALE < 1.0) or \
                 (now - getattr(app_callback, "_last_det", 0) < DETECTION_INTERVAL)
    frame = None
    if need_frame:
        fmt, w, h = get_caps_from_pad(pad)
        try:
            frame = get_numpy_from_buffer(buf, "RGB", w, h)
        except Exception:
            frame = gstbuffer_to_rgb(buf, w, h)
        if frame is not None and BUFFER_SCALE < 1.0:
            frame = cv2.resize(frame,
                               (int(w * BUFFER_SCALE), int(h * BUFFER_SCALE)),
                               interpolation=cv2.INTER_AREA)
    if frame is not None:
        with S.lock:
            S.frames.append(frame)
            S.stamps.append(now)

    # detecção (sub-amostrada)
    if now - getattr(app_callback, "_last_det", 0) >= DETECTION_INTERVAL:
        app_callback._last_det = now
        roi  = hailo.get_roi_from_buffer(buf)
        dets = roi.get_objects_typed(hailo.HAILO_DETECTION)
        crossed = False
        for d in dets:
            if d.get_label() != "person":
                continue
            lms = d.get_objects_typed(hailo.HAILO_LANDMARKS)
            if not lms:
                continue
            pts = lms[0].get_points()
            need = [0, 5, 6, 9, 10]  # nose, shoulders, wrists
            if any(pts[i].confidence() < 0.5 for i in need):
                continue
            fmt, w, h = get_caps_from_pad(pad)
            bb = d.get_bbox()
            def xy(p):
                return ((p.x() * bb.width() + bb.xmin()) * w,
                        (p.y() * bb.height() + bb.ymin()) * h)
            nose, ls, rs, lw, rw = map(xy, (pts[i] for i in need))
            crossed = lw[0] > rs[0] and rw[0] < ls[0] and \
                      lw[1] < nose[1] and rw[1] < nose[1]
            if crossed:
                break
        del roi, dets
        if crossed:
            if S.pose_start is None:
                S.pose_start = now
            elif now - S.pose_start >= POSE_HOLD_SECONDS and not S.pose_ready:
                S.pose_ready = True
                S.auto_cnt  += 1
                with S.lock:
                    S.trigger_fr = frame
                save_buffer(manual=False)
        else:
            S.pose_start = None
            S.pose_ready = False

    # GC leve
    if S.get_count() % 200 == 0:
        if psutil.Process().memory_info().rss / 1024 / 1024 > 700:
            gc.collect()
    return Gst.PadProbeReturn.OK

# ─────────────────── UI (thread secundária) ───────────────────────────────
def ui_loop(stop_event):
    cv2.startWindowThread()
    print("[TECLAS] q  g  +  -")
    while not stop_event.is_set():
        with S.lock:
            disp = S.frames[-1].copy() if S.frames else None
        if disp is None:
            time.sleep(0.01)
            continue
        h, w = disp.shape[:2]
        txt = f"Auto:{S.auto_cnt}  Man:{S.manual_cnt}  Buf:{len(S.frames)}/{S.frames.maxlen}"
        cv2.putText(disp, txt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if S.recording:
            cv2.putText(disp, "SALVANDO...",
                        (w // 2 - 120, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("Hailo Pose", cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            stop_event.set()
        elif k in (ord('+'), ord('=')):
            S.buf_sec += 5
        elif k in (ord('-'), ord('_')) and S.buf_sec > 5:
            S.buf_sec -= 5
        elif k == ord('g'):
            with S.lock:
                S.trigger_fr = disp
            save_buffer(manual=True)
            S.manual_cnt += 1
        time.sleep(0.003)
    cv2.destroyAllWindows()

# ─────────────────────────── MAIN ─────────────────────────────────────────
if __name__ == "__main__":
    print(f"[BOOT] buf={BUFFER_SECONDS}s det={1/DETECTION_INTERVAL:.1f}/s "
          f"scale={BUFFER_SCALE} codec=MJPG")
    Gst.init(None)
    app = GStreamerPoseEstimationApp(app_callback, S)

    stop_evt = threading.Event()
    # UI em *thread* separada
    threading.Thread(target=ui_loop, args=(stop_evt,), daemon=True).start()

    try:
        app.run()                 # BLOQUEANTE na main-thread (sem assert)
    finally:
        stop_evt.set()            # garante saída da UI
        if LED_AVAILABLE:
            _led.set_value(0);  _led.release();  _chip.close()
        print("[EXIT] finalizado")
