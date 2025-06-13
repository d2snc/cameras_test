# -*- coding: utf-8 -*-
"""
Detecção de braços cruzados utilizando Hailo‑8 + GStreamerPoseEstimationApp.

Principais diferenças em relação ao script YOLOv8 original:
• Toda a inferência de pose roda no Hailo‑8 via `pose_estimation_pipeline`, liberando CPU.
• Lógica de braços cruzados, buffer de gravação, controle de LED e hot‑keys foram preservados.

Pré‑requisitos (exemplo RPi 5 + Hailo‑8):
  pip install hailo-sdk-hailort hailo_apps_infra opencv-python numpy gpiod
  sudo apt install gstreamer1.0-plugins-{base,good,bad} libgpiod-dev

Execute:
  python3 arm_cross_detection_hailo.py
"""

import gi
import os
import cv2
import time
import threading
import numpy as np
from collections import deque
from datetime import datetime

# Hailo / GStreamer
import hailo
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

gi.require_version('Gst', '1.0')
from gi.repository import Gst

# ----------------------------------------------------------------------------------
# CONFIGURAÇÕES GERAIS
# ----------------------------------------------------------------------------------
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"  # evita glitches

BUFFER_SEGUNDOS = 50            # segundos armazenados no buffer circular
PASTA_GRAVACAO = "gravacoes"
PREFIXO_AUTO = "braco_cruzado_"
PREFIXO_MANUAL = "manual_"

DETECTION_INTERVAL = 0.2        # seg entre verificações de braço (≈5 Hz)
THRESHOLD_SEGS = 0.8            # seg necessários para confirmar braços cruzados

# GPIO – LED indicador (BCM‑17)
try:
    import gpiod
    LED_LINE = gpiod.Chip("gpiochip0").get_line(17)
    LED_LINE.request(consumer="hailo-led", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
    LED_OK = True
except Exception:
    LED_OK = False

# ----------------------------------------------------------------------------------
# COCO KEYPOINTS
# ----------------------------------------------------------------------------------
KP = {
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
LW, RW = KP['left_wrist'], KP['right_wrist']
LS, RS = KP['left_shoulder'], KP['right_shoulder']
LE, RE = KP['left_elbow'], KP['right_elbow']
NOSE = KP['nose']

# ----------------------------------------------------------------------------------
# UTILITÁRIOS
# ----------------------------------------------------------------------------------

def pulse_led(duration: float = 3.0):
    if not LED_OK:
        return
    LED_LINE.set_value(1)
    threading.Timer(duration, lambda: LED_LINE.set_value(0), daemon=True).start()

# ----------------------------------------------------------------------------------
# GRAVAÇÃO DE VÍDEO E BUFFER
# ----------------------------------------------------------------------------------
FPS_EST = 30.0
BUFFER_LOCK = threading.Lock()
FRAME_BUFFER: deque[np.ndarray] = deque(maxlen=int(FPS_EST * BUFFER_SEGUNDOS))
TIME_BUFFER: deque[float] = deque(maxlen=int(FPS_EST * BUFFER_SEGUNDOS))

gravando_video = False
contador_auto = 0
contador_manual = 0


def _video_writer_worker(frames, arquivo, fps):
    global gravando_video
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(arquivo, fourcc, fps, (w, h))
    if not vw.isOpened():
        print(f"[ERRO] Não abriu {arquivo}")
        gravando_video = False
        return
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"[OK] Salvo {arquivo} ({len(frames)}f @ {fps:.1f}fps)")
    gravando_video = False


def salvar_buffer(trigger_frame=None, manual=False):
    """Salva o conteúdo do buffer em AVI; opcionalmente inclui frame gatilho."""
    global gravando_video, contador_manual
    if gravando_video:
        return
    with BUFFER_LOCK:
        if len(FRAME_BUFFER) == 0:
            return
        frames = [f.copy() for f in FRAME_BUFFER]
    if trigger_frame is not None:
        frames.append(trigger_frame.copy())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = PREFIXO_MANUAL if manual else PREFIXO_AUTO
    if manual:
        contador_manual += 1
    nome = os.path.join(PASTA_GRAVACAO, f"{prefix}{ts}.avi")
    fps = max(10.0, min(FPS_EST, 30.0))
    gravando_video = True
    threading.Thread(target=_video_writer_worker, args=(frames, nome, fps), daemon=True).start()

# ----------------------------------------------------------------------------------
# USER DATA & CALLBACK
# ----------------------------------------------------------------------------------
class CrossArmsUserData(app_callback_class):
    def __init__(self):
        super().__init__()
        self.last_detect_t = 0.0
        self.cross_start_t: float | None = None
        self.show_full = True
        self.frame_out: np.ndarray | None = None
        self._lock = threading.Lock()

    def toggle_show(self):
        with self._lock:
            self.show_full = not self.show_full


USER = CrossArmsUserData()


def arms_crossed(points, w, h):
    try:
        lw = np.array([points[LW].x(), points[LW].y()]) * [w, h]
        rw = np.array([points[RW].x(), points[RW].y()]) * [w, h]
        ls = np.array([points[LS].x(), points[LS].y()]) * [w, h]
        rs = np.array([points[RS].x(), points[RS].y()]) * [w, h]
        ns = np.array([points[NOSE].x(), points[NOSE].y()]) * [w, h]
        crossed = lw[0] > rs[0] and rw[0] < ls[0] and lw[0] > rw[0]
        above_head = lw[1] < ns[1] and rw[1] < ns[1]
        close_x = abs(lw[0] - rw[0]) < w * 0.3
        return crossed and above_head and close_x
    except Exception:
        return False


def app_callback(pad, info, user_data: CrossArmsUserData):
    global FPS_EST, contador_auto
    buf = info.get_buffer()
    if buf is None:
        return Gst.PadProbeReturn.OK

    fmt, w, h = get_caps_from_pad(pad)
    frame = get_numpy_from_buffer(buf, fmt, w, h) if user_data.use_frame else None
    now = time.time()

    # FPS estimado
    user_data.increment()
    if user_data.get_count() == 10:
        dt = now - user_data.last_detect_t
        if dt > 0:
            FPS_EST = 10 / dt
        user_data.last_detect_t = now

    # Buffer circular
    if frame is not None:
        with BUFFER_LOCK:
            FRAME_BUFFER.append(frame.copy())
            TIME_BUFFER.append(now)

    # Rate‑limit
    if now - user_data.last_detect_t < DETECTION_INTERVAL:
        return Gst.PadProbeReturn.OK
    user_data.last_detect_t = now

    # Deteções Hailo
    roi = hailo.get_roi_from_buffer(buf)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    cruzado = False
    for det in detections:
        if det.get_label() != "person":
            continue
        lmarks = det.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not lmarks:
            continue
        pts = lmarks[0].get_points()
        if len(pts) < 17:
            continue
        if arms_crossed(pts, w, h):
            cruzado = True
            break

    # Lógica de disparo
    if cruzado:
        if user_data.cross_start_t is None:
            user_data.cross_start_t = now
        elif now - user_data.cross_start_t >= THRESHOLD_SEGS and not gravando_video:
            contador_auto += 1
            pulse_led(3)
            salvar_buffer(trigger_frame=frame, manual=False)
            user_data.cross_start_t = None
    else:
        user_data.cross_start_t = None

    # Overlay
    if frame is not None and user_data.show_full:
        cv2.putText(frame, f"Braços cruzados: {cruzado}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if cruzado else (0, 0, 255), 2)
        cv2.putText(frame, f"Auto: {contador_auto}  Manual: {contador_manual}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        user_data.set_frame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    return Gst.PadProbeReturn.OK

# ----------------------------------------------------------------------------------
# DISPLAY & TECLADO
# ----------------------------------------------------------------------------------

def display_loop():
    global BUFFER_SEGUNDOS
    print("Comandos: q=Quit | g=Gravar manual | v=Alterna visualização | +/- ajusta buffer")
    while True:
        frame = USER.get_frame()
        if frame is not None:
            cv2.imshow("Detecção Braços Cruzados – Hailo", frame)
        else:
            time.sleep(0.05)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            salvar_buffer(trigger_frame=frame, manual=True)
        elif key == ord('v'):
            USER.toggle_show()
        elif key in (ord('+'), ord('=')):
            BUFFER_SEGUNDOS += 5
            with BUFFER_LOCK:
                FRAME_BUFFER = deque(FRAME_BUFFER, maxlen=int(FPS_EST * BUFFER_SEGUNDOS))
                TIME_BUFFER = deque(TIME_BUFFER, maxlen=int(FPS_EST * BUFFER_SEGUNDOS))
            print(f"Buffer: {BUFFER_SEGUNDOS}s")
        elif key in (ord('-'), ord('_')) and BUFFER_SEGUNDOS > 5:
            BUFFER_SEGUNDOS -= 5
            with BUFFER_LOCK:
                FRAME_BUFFER = deque(FRAME_BUFFER, maxlen=int(FPS_EST * BUFFER_SEGUNDOS))
                TIME_BUFFER = deque(TIME_BUFFER, maxlen=int(FPS_EST * BUFFER_SEGUNDOS))
            print(f"Buffer: {BUFFER_SEGUNDOS}s")
    cv2.destroyAllWindows()

# ----------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------
if __
