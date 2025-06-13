#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pose-estimation com detecção de “braços cruzados acima da cabeça”.
• Buffer de 40 s  • Salva 30 s antes + 5 s depois do evento
• Grava H.264 MP4 em processo separado (sem travar o pipeline)
• LED (GPIO-17) pisca a cada 0,75 s enquanto o arquivo é escrito
Douglas Lima – 2025-06-13
"""

import gi, os, time, cv2, numpy as np, hailo, threading, multiprocessing as mp
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from collections import deque
from datetime import datetime
from queue import Full as QFull

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# ─────────────────────────── LED (GPIO-17) ───────────────────────────
try:
    import gpiod
    _chip = gpiod.Chip("gpiochip0")
    _led  = _chip.get_line(17)
    _led.request(consumer="pose-led", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
    LED_OK = True
    print("GPIO17 LED pronto")
except Exception as e:
    LED_OK = False
    print("GPIO17 LED desativado:", e)

def pulse_led(dur=0.15):
    if not LED_OK:
        return
    _led.set_value(1)
    threading.Timer(dur, lambda: _led.set_value(0), daemon=True).start()

def _blink_led(stop_evt, on_ms=150, per_ms=750):
    while not stop_evt.is_set():
        pulse_led(on_ms / 1000)
        stop_evt.wait(per_ms / 1000)

# ──────────────────────── Pasta de gravações ─────────────────────────
SAVE_DIR = "gravacoes"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────── Processo gravador MP4 ───────────────────────
def _choose_encoder(fps):
    """Escolhe o primeiro encoder H.264 disponível."""
    for enc in ("v4l2h264enc", "omxh264enc", "x264enc"):
        if Gst.ElementFactory.find(enc):
            if enc == "v4l2h264enc":
                return f'v4l2h264enc extra-controls="encode,frame_level_rate_control_enable=true" keyframe-period={fps*2}'
            if enc == "omxh264enc":
                return 'omxh264enc control-rate=variable target-bitrate=2000000'
            return 'x264enc tune=zerolatency speed-preset=veryfast'
    return None

def saver_proc(f_q: mp.Queue, stop_evt: mp.Event, w, h, fps):
    Gst.init(None)                         # precisa inicializar no filho
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(SAVE_DIR, f"arms_crossed_{ts}.mp4")
    print(f"[Saver] gravando {fname}")

    enc_str = _choose_encoder(fps)
    use_gst = False
    if enc_str:
        pipe = (
            f"gst-launch-1.0 -q appsrc name=src is-live=true block=true "
            f"format=3 do-timestamp=true ! jpegdec ! "
            f"videoconvert ! video/x-raw,format=I420,width={w},height={h},framerate={fps}/1 ! "
            f"{enc_str} ! h264parse ! mp4mux ! filesink location={fname}"
        )
        try:
            import subprocess, shlex
            gstp = subprocess.Popen(shlex.split(pipe), stdin=subprocess.PIPE)
            use_gst = True
        except Exception as e:
            print("[Saver] Falha GStreamer, usando OpenCV:", e)

    if not use_gst:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(fname, fourcc, fps, (w, h))

    frames = 0
    while not (stop_evt.is_set() and f_q.empty()):
        try:
            jpeg = f_q.get(timeout=0.1)
        except Exception:
            continue
        if use_gst:
            gstp.stdin.write(jpeg)
        else:
            img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            vw.write(img)
        frames += 1

    if use_gst:
        gstp.stdin.close()
        gstp.wait()
    else:
        vw.release()
    print(f"[Saver] concluído – {frames} quadros salvos")

# ─────────────────────── Utilidades de detecção ─────────────────────
KP = {"nose": 0, "left_wrist": 9, "right_wrist": 10}
def arms_crossed(points, bbox, W, H):
    try:
        lw, rw, nz = (points[KP[k]] for k in ("left_wrist","right_wrist","nose"))
        lwx = (lw.x()*bbox.width()+bbox.xmin())*W
        lwy = (lw.y()*bbox.height()+bbox.ymin())*H
        rwx = (rw.x()*bbox.width()+bbox.xmin())*W
        rwy = (rw.y()*bbox.height()+bbox.ymin())*H
        nzy = (nz.y()*bbox.height()+bbox.ymin())*H
        return (lwy < nzy and rwy < nzy) and (lwx > rwx)
    except Exception:
        return False

# ───────────────────────── Classe de callback ───────────────────────
class UserCB(app_callback_class):
    FPS, PRE_S, POST_S = 30, 30, 5
    COOLDOWN = 5
    def __init__(self):
        super().__init__()
        self.buf = deque(maxlen=self.FPS*(self.PRE_S+self.POST_S+5))
        self.trigger, self.after = False, 0
        self.last_evt, self.total = 0.0, 0
        self.dim = None
        # gravador
        self.f_q = self.stop_evt = self.saver = None
        self.blink_evt = None
        self.status = "Arms Not Crossed"

    # ─────────── iniciar gravador ───────────
    def _start_saver(self, h, w):
        self.f_q = mp.Queue(maxsize=500)
        self.stop_evt = mp.Event()
        self.saver = mp.Process(target=saver_proc,
                                args=(self.f_q, self.stop_evt, w, h, self.FPS),
                                daemon=True)
        self.saver.start()
        # LED blink
        self.blink_evt = threading.Event()
        threading.Thread(target=_blink_led, args=(self.blink_evt,),
                         daemon=True).start()

    # ─────────── enviar quadro ──────────────
    def _send(self, frame):
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY),85])
        if ok and self.saver and self.saver.is_alive():
            try: self.f_q.put_nowait(jpg.tobytes())
            except QFull: pass

# ─────────────────────── Pad-probe da aplicação ─────────────────────
def app_cb(pad, info, user: UserCB):
    buf = info.get_buffer()
    fmt,W,H = get_caps_from_pad(pad)
    if buf is None or fmt is None: return Gst.PadProbeReturn.OK
    rgb = get_numpy_from_buffer(buf, fmt, W, H)

    # -------- detecção -----------------------
    crossed_now = False
    roi = hailo.get_roi_from_buffer(buf)
    for det in roi.get_objects_typed(hailo.HAILO_DETECTION):
        if det.get_label()!="person": continue
        lms = det.get_objects_typed(hailo.HAILO_LANDMARKS)
        if lms and arms_crossed(lms[0].get_points(), det.get_bbox(), W, H):
            crossed_now = True; break

    t = time.time()
    if crossed_now:
        user.status = "Arms Crossed!"
        if not user.trigger and t-user.last_evt>user.COOLDOWN:
            user.trigger, user.after = True, 0
            user.last_evt = t; user.total += 1
            pulse_led(0.3)
            print("*** braços cruzados – capturando ***")
    else:
        user.status = "Arms Not Crossed"

    # pós-evento
    if user.trigger:
        user.after += 1
        if user.after >= user.FPS*user.POST_S:
            user.trigger = False
            user.stop_evt.set()
            if user.blink_evt: user.blink_evt.set()
            print("*** captura concluída – salvando ***")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr,user.status,(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.1,
                (0,0,255) if "Crossed" in user.status else (0,255,0),2)
    cv2.putText(bgr,f"Detections: {user.total}",(10,70),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),1)

    # buffer & gravação
    user.buf.append(bgr)
    if user.trigger and user.after==1 and not user.saver:
        h,w = bgr.shape[:2]; user.dim=(h,w); user._start_saver(h,w)
        for f in list(user.buf)[-user.FPS*user.PRE_S:]:
            user._send(f)
    elif user.saver and user.saver.is_alive():
        user._send(bgr)

    user.set_frame(bgr)
    return Gst.PadProbeReturn.OK

# ──────────────────────────── Main ────────────────────────────────
if __name__ == "__main__":
    Gst.init(None)
    print("Pose-detector ativo – Ctrl-C para sair")
    try:
        app = GStreamerPoseEstimationApp(app_cb, UserCB())
        app.run()
    finally:
        if LED_OK: _led.set_value(0)
