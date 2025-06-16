#!/usr/bin/env python3
"""
Arms‑crossed detector + rolling video recorder accelerated by Hailo‑8L.
2025‑06‑16 rev.5 — visualização sempre ativa
• Janela "Pose" com overlay ativada por padrão (OpenCV).
• Novo flag `--no-show` para rodar headless se necessário.
• Mantém Picamera2 padrão; `--usb` muda para webcam.
• HEF padrão `yolov8s_pose.hef`.
"""
import argparse, collections, os, pathlib, queue, threading
from datetime import datetime

import gi, cv2, numpy as np, hailo

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from hailo_apps_infra.hailo_rpi_common import (
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# --------------------------- GPIO ------------------------------------------
try:
    import gpiod
    LED = gpiod.Chip("gpiochip0").get_line(17)
    LED.request(consumer="pose-led", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
    GPIO_OK = True
except Exception as e:
    print("[WARN] GPIO17 LED disabled:", e)
    GPIO_OK = False

def pulse_led(dur=3.0):
    if GPIO_OK:
        LED.set_value(1)
        threading.Timer(dur, lambda: LED.set_value(0), daemon=True).start()

# ---------------- Heurística braços cruzados -------------------------------
class ArmsCrossDetector:
    NOSE, L_WRIST, R_WRIST = 0, 9, 10
    def __init__(self, margin=10): self.margin = margin
    def __call__(self, k):
        try:
            nose_y = k[self.NOSE][1]; lw_x, lw_y = k[self.L_WRIST]; rw_x, rw_y = k[self.R_WRIST]
            return lw_y < nose_y-self.margin and rw_y < nose_y-self.margin and lw_x > rw_x
        except Exception:
            return False

# -------------------------- Callback ---------------------------------------
class ArmsCallback(app_callback_class):
    def __init__(self, buf_len, fps, preview_q):
        super().__init__(); self.use_frame = True
        self.buf = collections.deque(maxlen=buf_len); self.fps = fps; self.det = ArmsCrossDetector()
        self.preview_q = preview_q; self.cool = 0
    def _save(self):
        if not self.buf: return
        h,w,_ = self.buf[0].shape; ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        out=pathlib.Path("recordings"); out.mkdir(exist_ok=True)
        path=out/f"arms_{ts}.mp4"; vw=cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),self.fps,(w,h))
        for f in list(self.buf): vw.write(f)
        vw.release(); print(f"[SAVE] {path} ({len(self.buf)}f)")
    def handle_buffer(self, pad, info):
        buf=info.get_buffer(); frame=get_numpy_from_buffer(buf); self.buf.append(frame.copy())
        kpts=[]; meta=buf.get_meta("HailoJsonMeta")
        if meta:
            import json
            objs=json.loads(meta.get_json()).get("objects",[])
            if objs:
                kpts=[(p['x'],p['y']) for p in max(objs,key=lambda o:o['confidence'])['keypoints']]
        crossed=self.det(kpts)
        label="ARMS CROSSED" if crossed else "ARMS NOT CROSSED"
        cv2.putText(frame, label, (30,40), cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255) if crossed else (0,255,0),3)
        if self.preview_q:
            try: self.preview_q.put_nowait(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            except queue.Full: pass
        if crossed and self.cool==0:
            pulse_led(); threading.Thread(target=self._save, daemon=True).start(); self.cool=self.fps*5
        else: self.cool=max(0,self.cool-1)
        return Gst.PadProbeReturn.OK

# -------------------------- Preview Thread ---------------------------------
class PreviewThread(threading.Thread):
    def __init__(self,q): super().__init__(daemon=True); self.q=q; self._stop=threading.Event()
    def run(self):
        while not self._stop.is_set():
            try:
                frame=self.q.get(timeout=0.1)
                cv2.imshow("Pose", frame)
            except queue.Empty:
                continue
            if cv2.waitKey(1)&0xFF==27: self._stop.set()
        cv2.destroyAllWindows()
    def stop(self): self._stop.set()

# --------------------------- main ------------------------------------------

def main():
    Gst.init(None)
    ap=argparse.ArgumentParser()
    ap.add_argument("--hef", default="yolov8s_pose.hef", help="HEF file (default: yolov8s_pose.hef)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--buffer", type=int, default=20)
    ap.add_argument("--no-show", action="store_true", help="Roda sem janela de preview")
    ap.add_argument("--usb", action="store_true", help="/dev/video0 webcam instead of Picamera2")
    args=ap.parse_args()

    if not os.path.exists(args.hef):
        raise FileNotFoundError(f"HEF '{args.hef}' não encontrado")

    preview_q = None if args.no_show else queue.Queue(maxsize=2)
    user_data = ArmsCallback(args.buffer*args.fps, args.fps, preview_q)

    def cb(pad, info, ud): return ud.handle_buffer(pad, info)
    app = GStreamerPoseEstimationApp(app_callback=cb, user_data=user_data)
    app.hef_path=os.path.abspath(args.hef)
    app.src_caps=f"video/x-raw,format=RGB,width={args.width},height={args.height},framerate={args.fps}/1"
    app.video_source="v4l2src device=/dev/video0" if args.usb else "libcamerasrc"

    pv=PreviewThread(preview_q) if preview_q else None
    if pv: pv.start()
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop();
        if pv:
            pv.stop(); pv.join()

if __name__=="__main__":
    main()
