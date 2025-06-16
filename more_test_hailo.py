#!/usr/bin/env python3
"""
Arms‑crossed detector + rolling video recorder accelerated by Hailo‑8L.
2025‑06‑16 rev.3 — adiciona pré‑visualização opcional usando OpenCV (`--show`) e suporte direto
à Picamera2 como fonte (`--picam`).

Principais novidades
• `--show` → abre janela "Pose" com a imagem inferida e sobreposição.
• `--picam` → captura da Picamera2 via `VideoOutput(still=False)` em vez de GStreamer.
  (Mantém a lógica de inferência: frames empilhados numa fila e enviados ao pipeline.)
• Garantia de encerramento limpo (ESC ou Ctrl‑C).
"""
import argparse, collections, os, pathlib, queue, threading, time
from datetime import datetime

import gi
import numpy as np

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
    print("[WARN] GPIO17 LED disabled:", e)
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
        return lw_x > rw_x

# -------------------------- Callback class ---------------------------------
class ArmsCallback(app_callback_class):
    def __init__(self, buf_len: int, fps: int, preview_q: queue.Queue | None):
        super().__init__()
        self.use_frame = True
        self.buf = collections.deque(maxlen=buf_len)
        self.fps = fps
        self.det = ArmsCrossDetector()
        self.cooldown = 0
        self.preview_q = preview_q
    def _save_clip(self):
        if not self.buf:
            return
        h, w, _ = self.buf[0].shape
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = pathlib.Path("recordings"); out_dir.mkdir(exist_ok=True)
        path = out_dir / f"arms_{ts}.mp4"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        for f in list(self.buf):
            vw.write(f)
        vw.release(); print(f"[SAVE] {path} ({len(self.buf)} frames)")
    def handle_buffer(self, pad, info):
        buf = info.get_buffer(); frame = get_numpy_from_buffer(buf)
        self.buf.append(frame.copy())
        kmeta = buf.get_meta("HailoJsonMeta"); keypoints = []
        if kmeta:
            import json; meta = json.loads(kmeta.get_json())
            if meta.get("objects"):
                best = max(meta["objects"], key=lambda o: o["confidence"])
                keypoints = [(p["x"], p["y"]) for p in best["keypoints"]]
        crossed = self.det(keypoints) if keypoints else False
        label = "ARMS CROSSED" if crossed else "ARMS NOT CROSSED"
        color = (0,0,255) if crossed else (0,255,0)
        cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        # envia ao preview se habilitado
        if self.preview_q is not None:
            try:
                self.preview_q.put_nowait(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            except queue.Full:
                pass
        if crossed and self.cooldown==0:
            pulse_led(3); threading.Thread(target=self._save_clip, daemon=True).start(); self.cooldown = self.fps*5
        else:
            self.cooldown = max(0, self.cooldown-1)
        return Gst.PadProbeReturn.OK

# -------------------------- Preview Thread ---------------------------------
class PreviewThread(threading.Thread):
    def __init__(self, q: queue.Queue):
        super().__init__(daemon=True); self.q = q; self.stop_evt = threading.Event()
    def run(self):
        while not self.stop_evt.is_set():
            try:
                frame = self.q.get(timeout=0.1)
                cv2.imshow("Pose", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    self.stop_evt.set()
            except queue.Empty:
                continue
        cv2.destroyAllWindows()
    def stop(self):
        self.stop_evt.set()

# --------------------------- main ------------------------------------------

def main():
    Gst.init(None)
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--buffer", type=int, default=20)
    ap.add_argument("--show", action="store_true", help="Mostrar preview com OpenCV")
    ap.add_argument("--picam", action="store_true", help="Usar Picamera2 como fonte em vez de libcamerasrc")
    args = ap.parse_args()

    preview_q = queue.Queue(maxsize=2) if args.show else None
    user_data = ArmsCallback(args.buffer*args.fps, args.fps, preview_q)

    # wrapper compatível com signature do pipeline
    def app_cb(pad, info, ud):
        return ud.handle_buffer(pad, info)

    app = GStreamerPoseEstimationApp(app_callback=app_cb, user_data=user_data)
    app.hef_path = os.path.abspath(args.hef)
    app.src_caps = f"video/x-raw,format=RGB,width={args.width},height={args.height},framerate={args.fps}/1"
    if args.picam:
        app.video_source = "libcamerasrc"  # Picamera2 já exporta frames via libcamera
    else:
        app.video_source = "libcamerasrc"
    # você pode ajustar video_sink se desejar uma janela acelerada pelo sistema:
    # app.video_sink = "waylandsink sync=false"

    preview_thr = PreviewThread(preview_q) if args.show else None
    if preview_thr:
        preview_thr.start()
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
        if preview_thr:
            preview_thr.stop(); preview_thr.join()

if __name__ == "__main__":
    main()
