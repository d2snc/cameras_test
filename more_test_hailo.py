#!/usr/bin/env python3
"""
Arms‑crossed detector + rolling video recorder accelerated by Hailo‑8L.
Fix 2025‑06‑16 rev.2 — corrige AttributeError 'NoneType' object has no attribute 'use_frame'
Key changes ★ Passamos o objeto user_data (que herda app_callback_class) ao GStreamerPoseEstimationApp e
marcamos `self.use_frame = True` no construtor, seguindo o padrão dos exemplos oficiais.
"""
import argparse, collections, os, pathlib, threading, time
from datetime import datetime
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import cv2
import hailo

from hailo_apps_infra.hailo_rpi_common import (
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# --------------------------- GPIO ------------------------------------------
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
    if not GPIO_OK:
        return
    LED.set_value(1)
    t = threading.Timer(duration, lambda: LED.set_value(0))
    t.daemon = True
    t.start()

# ------------------ Heurística braços cruzados -----------------------------
class ArmsCrossDetector:
    NOSE, L_WRIST, R_WRIST = 0, 9, 10
    def __init__(self, margin: int = 10):
        self.margin = margin
    def __call__(self, kpts):
        try:
            nose_y = kpts[self.NOSE][1]
            lw_x, lw_y = kpts[self.L_WRIST]
            rw_x, rw_y = kpts[self.R_WRIST]
        except Exception:
            return False
        if not (lw_y < nose_y - self.margin and rw_y < nose_y - self.margin):
            return False
        return lw_x > rw_x  # inversão ↔ braços cruzados

# -------------------------- Callback class ---------------------------------
class ArmsCallback(app_callback_class):
    def __init__(self, buf_len: int, fps: int):
        super().__init__()
        self.use_frame = True  # *** chave p/ evitar AttributeError ***
        self.buf = collections.deque(maxlen=buf_len)
        self.fps = fps
        self.det = ArmsCrossDetector()
        self.cooldown = 0
    # util p/ gravação
    def _save_clip(self):
        if not self.buf:
            return
        h, w, _ = self.buf[0].shape
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = pathlib.Path("recordings")
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"arms_{ts}.mp4"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        for f in list(self.buf):
            vw.write(f)
        vw.release()
        print(f"[SAVE] {path} ({len(self.buf)} frames)")

    # método chamado pelo functions wrapper
    def handle_buffer(self, pad, info):
        buf = info.get_buffer()
        frame = get_numpy_from_buffer(buf)
        self.buf.append(frame.copy())
        kmeta = buf.get_meta("HailoJsonMeta")
        keypoints = []
        if kmeta:
            import json
            meta = json.loads(kmeta.get_json())
            if meta.get("objects"):
                best = max(meta["objects"], key=lambda o: o["confidence"])
                keypoints = [(p["x"], p["y"]) for p in best["keypoints"]]
        crossed = self.det(keypoints) if keypoints else False
        if crossed and self.cooldown == 0:
            pulse_led(3)
            threading.Thread(target=self._save_clip, daemon=True).start()
            self.cooldown = self.fps * 5
        else:
            self.cooldown = max(0, self.cooldown - 1)
        return Gst.PadProbeReturn.OK

# --------------------------- main ------------------------------------------

def main():
    Gst.init(None)
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--buffer", type=int, default=20, help="segundos no buffer")
    args = ap.parse_args()

    user_data = ArmsCallback(args.buffer * args.fps, args.fps)

    # wrapper → delega ao método acima, preservando assinatura esperada
    def app_cb(pad, info, ud):
        return ud.handle_buffer(pad, info)

    app = GStreamerPoseEstimationApp(app_callback=app_cb, user_data=user_data)
    app.hef_path = os.path.abspath(args.hef)
    app.src_caps = f"video/x-raw,format=RGB,width={args.width},height={args.height},framerate={args.fps}/1"

    try:
        app.run()
    except KeyboardInterrupt:
        app.stop()

if __name__ == "__main__":
    main()
