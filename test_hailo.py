# -*- coding: utf-8 -*-
"""
Detecção de braços cruzados utilizando Hailo‑8 + GStreamerPoseEstimationApp.

Principais diferenças em relação ao script YOLOv8 original:
• O backbone de inferência é executado inteiramente no Hailo‑8 mediante o pipeline
  `hailo.apps.pose_estimation_pipeline`, garantindo FPS elevado com baixo uso da CPU.
• A calibração de pontos‑chave (keypoints) permanece compatível com COCO; o cálculo de
  braços cruzados segue a mesma lógica original.
• O gerenciamento de buffer circular, gravação de vídeo (automática e manual), LED GPIO
  (via libgpiod) e interface de teclado continuam praticamente idênticos.

Pré‑requisitos:
  pip install hailo‑sdk‑hailort hailo.apps‑infra opencv‑python numpy gpiod
  # Demais dependências (GStreamer, libgpiod‑dev, etc.) devem estar presentes

Execute com:
  python3 arm_cross_detection_hailo.py
"""

import gi
import os
import cv2
import time
import json
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
from gi.repository import Gst, GLib

# ----------------------------------------------------------------------------------
# CONFIGURAÇÕES
# ----------------------------------------------------------------------------------
# Áudio OFF; surge ruído caso openCV Vá chamar cv2.imshow em thread diferente
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

BUFFER_SEGUNDOS = 50            # Quantos segundos manter no buffer (circular)
PASTA_GRAVACAO = "gravacoes"    # Pasta onde vídeos serão salvos
PREFIXO_AUTO = "braco_cruzado_"
PREFIXO_MANUAL = "manual_"

DETECTION_INTERVAL = 0.2        # 5 detecções/s (reduz sobrecarga)
THRESHOLD_SEGS = 0.8            # Braços cruzados precisam durar >= 0.8 s

# GPIO – LED indicativo (pino BCM‑17)
try:
    import gpiod
    CHIP = gpiod.Chip("gpiochip0")
    LED_LINE = CHIP.get_line(17)
    LED_LINE.request(consumer="hailo‑led", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
    LED_OK = True
except Exception:
    LED_OK = False

# -------t---------------------------------------------------------------------------
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
# Para facilitar acesso rápido:
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
# GRAVAÇÃO DE VÍDEO
# ----------------------------------------------------------------------------------
FPS_EST = 30.0   # estimativa inicial
BUFFER_LOCK = threading.Lock()
FRAME_BUFFER: deque[np.ndarray] = deque(maxlen=int(FPS_EST * BUFFER_SEGUNDOS))
TIME_BUFFER: deque[float] = deque(maxlen=int(FPS_EST * BUFFER_SEGUNDOS))

gravando_video = False
contador_auto = 0
contador_manual = 0


def _video_writer_worker(frames, nome_arquivo, fps):
    global gravando_video
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(nome_arquivo, fourcc, fps, (w, h))
    if not vw.isOpened():
        print(f"[ERRO] Não foi possível abrir {nome_arquivo}")
        gravando_video = False
        return
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"[OK] Vídeo salvo: {nome_arquivo} — {len(frames)} frames @ {fps:.1f} fps")
    gravando_video = False


def salvar_buffer(trigger_frame=None, manual=False):
    global gravando_video, contador_manual
    if gravando_video:
        return
    with BUFFER_LOCK:
        if len(FRAME_BUFFER) == 0:
            return
        frames = [f.copy() for f in FRAME_BUFFER]
    if trigger_frame is not None:
        frames.append(trigger_frame.copy())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = PREFIXO_MANUAL if manual else PREFIXO_AUTO
    if manual:
        contador_manual += 1
    nome = os.path.join(PASTA_GRAVACAO, f"{prefix}{timestamp}.avi")
    fps = max(10.0, min(FPS_EST, 30.0))
    gravando_video = True
    threading.Thread(target=_video_writer_worker, args=(frames, nome, fps), daemon=True).start()

# ----------------------------------------------------------------------------------
# CALLBACK & USER DATA
# ----------------------------------------------------------------------------------
class CrossArmsUserData(app_callback_class):
    def __init__(self):
        super().__init__()
        self.last_detect_t = 0.0
        self.cross_start_t: float | None = None
        self.is_recording_flag = False
        self.show_full = True   # alterna modo de visualização
        self.show_diag = True
        self.frame_out: np.ndarray | None = None
        # Thread‑safe flags
        self._lock = threading.Lock()

    # Interface simples para alternar flags a partir da UI principal
    def toggle_show(self):
        with self._lock:
            self.show_full = not self.show_full

    def toggle_diag(self):
        with self._lock:
            self.show_diag = not self.show_diag


USER = CrossArmsUserData()


def arms_crossed_logic(points, w, h):
    """Recebe keypoints normalizados (0‑1) e retorna bool se braços cruzados."""
    try:
        lw = np.array([points[LW].x(), points[LW].y()]) * [w, h]
        rw = np.array([points[RW].x(), points[RW].y()]) * [w, h]
        ls = np.array([points[LS].x(), points[LS].y()]) * [w, h]
        rs = np.array([points[RS].x(), points[RS].y()]) * [w, h]
        le = np.array([points[LE].x(), points[LE].y()]) * [w, h]
        re = np.array([points[RE].x(), points[RE].y()]) * [w, h]
        ns = np.array([points[NOSE].x(), points[NOSE].y()]) * [w, h]
        wrists_crossed = (lw[0] > rs[0] and rw[0] < ls[0] and lw[0] > rw[0])
        arms_above_head = (lw[1] < ns[1] and rw[1] < ns[1])
        close_horiz = abs(lw[0] - rw[0]) < w * 0.3
        return wrists_crossed and arms_above_head and close_horiz
    except Exception:
        return False


# Core callback

def app_callback(pad, info, user_data: CrossArmsUserData):
    global FPS_EST, FRAME_BUFFER, TIME_BUFFER, contador_auto
    buf = info.get_buffer()
    if buf is None:
        return Gst.PadProbeReturn.OK

    # Frame extraction (RGB888) & timing --------------------------------------
    fmt, width, height = get_caps_from_pad(pad)
    frame = get_numpy_from_buffer(buf, fmt, width, height) if user_data.use_frame else None
    now = time.time()

    # Update FPS estimate
    user_data.increment()
    if user_data.get_count() == 10:
        dt = now - user_data.last_detect_t
        if dt > 0:
            FPS_EST = 10 / dt
        user_data.last_detect_t = now

    # Save in buffer (thread‑safe)
    if frame is not None:
        with BUFFER_LOCK:
            FRAME_BUFFER.append(frame.copy())
            TIME_BUFFER.append(now)

    # Rate‑limit arm detection to DETECTION_INTERVAL -------------------------
    if now - user_data.last_detect_t < DETECTION_INTERVAL:
        return Gst.PadProbeReturn.OK
    user_data.last_detect_t = now

    # Obtain Hailo detections --------------------------------------------------
    roi = hailo.get_roi_from_buffer(buf)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    algum_cruzado = False
    for det in detections:
        if det.get_label() != "person":
            continue
        lmarks = det.get_objects_typed(hailo.HAILO_LANDMARKS)
        if len(lmarks) == 0:
            continue
        pts = lmarks[0].get_points()  # list[ hailo.Point ]
        if len(pts) < 17:
            continue
        if arms_crossed_logic(pts, width, height):
            algum_cruzado = True
            break  # basta 1 pessoa

    # --- Temporização de braços cruzados ------------------------------------
    if algum_cruzado:
        if user_data.cross_start_t is None:
            user_data.cross_start_t = now
        elif now - user_data.cross_start_t >= THRESHOLD_SEGS and not gravando_video:
            contador_auto += 1
            pulse_led(3.0)
            salvar_buffer(trigger_frame=frame, manual=False)
            user_data.cross_start_t = None  # evita múltiplas gravações
    else:
        user_data.cross_start_t = None

    # Overlay & visualização --------------------------------------------------
    if frame is not None and user_data.show_full:
        cv2.putText(frame, f"Braços cruzados: {algum_cruzado}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if algum_cruzado else (0, 0, 255), 2)
        cv2.putText(frame, f"Auto: {contador_auto}  Manual: {contador_manual}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        user_data.set_frame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    return Gst.PadProbeReturn.OK

# ----------------------------------------------------------------------------------
# LOOP PRINCIPAL (display e interação de teclado)
# ----------------------------------------------------------------------------------

def display_loop():
    global BUFFER_SEGUNDOS
    print("\nComandos: q=Quit | g=Gravar manual | v=Alterna visualização | d=Diag | +/- ajusta buffer")
    while True:
        frame = USER.get_frame(timeout=0.05)  # bloqueia 50 ms
        if frame is not None:
            cv2.imshow("Detecção Braços Cruzados – Hailo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            salvar_buffer(trigger_frame=frame, manual=True)
        elif key == ord('v'):
            USER.toggle_show()
        elif key == ord('d'):
            USER.toggle_diag()
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
if __name__ == "__main__":
    if not os.path.exists(PASTA_GRAVACAO):
        os.makedirs(PASTA_GRAVACAO)

    app = GStreamerPoseEstimationApp(app_callback, USER)

    # Inicia GStreamer (em thread própria) --------------------------------------------------
    gst_thread = threading.Thread(target=app.run, daemon=True)
    gst_thread.start()

    # Loop de exibição / teclado -----------------------------------------------------------
    display_loop()

    # Encerramento ordenado ---------------------------------------------------------------
    app.stop()
    gst_thread.join(timeout=2)
    print("Programa finalizado.")
