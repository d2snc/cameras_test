#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi 5 + Hailo-8L
Detecção de braços cruzados acima da cabeça com YOLOv8-Pose
→ grava os 20 s anteriores em MP4
→ acende o LED na GPIO-17 durante 3 s
2025-06-16
"""
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import hailo
import cv2, numpy as np, threading, time, os
from collections import deque
from datetime import datetime
# ————————————————————————————————————————————
# GPIO 17 (LED) via libgpiod
# ————————————————————————————————————————————
try:
    import gpiod
    _chip = gpiod.Chip("gpiochip0")
    _led  = _chip.get_line(17)
    _led.request(consumer="pose-led", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
    LED_OK = True
except Exception as e:
    print("GPIO17 LED disabled:", e)
    LED_OK = False

def pulse_led(duration=3):
    """Liga o LED e desliga depois de <duration> s (thread não-bloqueante)."""
    if not LED_OK:
        return
    _led.set_value(1)
    threading.Timer(duration, lambda: _led.set_value(0)).start()

# ————————————————————————————————————————————
# Vídeo – parâmetros do buffer
# ————————————————————————————————————————————
BUFFER_SEC   = 20           # segundos de vídeo a manter em RAM
FPS_ESTIMADO = 30           # usado só para dimensionar o deque
MAX_FRAMES   = BUFFER_SEC * FPS_ESTIMADO
FRAME_BUFFER = deque(maxlen=MAX_FRAMES)

SAVE_DIR = "gravacoes"
os.makedirs(SAVE_DIR, exist_ok=True)

# ————————————————————————————————————————————
# Keypoints COCO usados pela YOLOv8-Pose
# (ordem definida no modelo pré-treinado — ver guia da Hailo) :contentReference[oaicite:0]{index=0}
# ————————————————————————————————————————————
KP = {
    "nose": 0,
    "left_eye": 1,  "right_eye": 2,
    "left_ear": 3,  "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7,    "right_elbow": 8,
    "left_wrist": 9,    "right_wrist": 10,
    "left_hip": 11,     "right_hip": 12,
    "left_knee": 13,    "right_knee": 14,
    "left_ankle": 15,   "right_ankle": 16,
}

# ————————————————————————————————————————————
# Função utilitária: decide se cruzou braços acima da cabeça
# ————————————————————————————————————————————
def crossed_above_head(points):
    """
    Recebe um dicionário {nome:(x_norm,y_norm)}, retorna True/False.
    Critério simples:
      1. Ambos pulsos (wrists) Y < nariz Y  (acima da cabeça).
      2. Distância euclidiana entre pulsos < 15 % da largura (estão próximos → cruzados).
    """
    nose   = points.get("nose")
    lwrist = points.get("left_wrist")
    rwrist = points.get("right_wrist")
    if None in (nose, lwrist, rwrist):
        return False
    # todos os Y menores (lembrando: YOLO usa coords normalizadas, origem canto sup-esq)
    above = lwrist[1] < nose[1] and rwrist[1] < nose[1]
    # proximidade horizontal/vertical
    dist = np.linalg.norm(np.array(lwrist) - np.array(rwrist))
    close = dist < 0.15     # limiar empírico
    return above and close

# ————————————————————————————————————————————
# Callback da aplicação GStreamer
# ————————————————————————————————————————————
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

class SharedState:
    """Mantém a última detecção e gerencia eventos de gravação."""
    def __init__(self):
        self.lock     = threading.Lock()
        self.last_pts = None    # dict keypoints
        self.width    = None
        self.height   = None
        self.recording = False

STATE = SharedState()

class UserCallback(app_callback_class):
    def __init__(self):
        super().__init__()

def app_callback(pad, info, user_data):
    """Executa a cada frame: guarda quadro, checa pose e dispara ações."""
    buf = info.get_buffer()
    if buf is None:
        return Gst.PadProbeReturn.OK

    # ——— guardar frame no buffer circular ———
    frame = get_numpy_from_buffer(buf)      # RGB888
    FRAME_BUFFER.append(frame.copy())

    # ——— decodificar detecções do Hailo ———
    roi = hailo.get_roi_from_buffer(buf)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # buscar a pessoa – YOLOv8-Pose tem label "person"
    person_det = next((d for d in detections if d.get_label() == "person"), None)
    if person_det:
        lm = person_det.get_objects_typed(hailo.HAILO_LANDMARKS)
        if lm:
            pts = lm[0].get_points()
            coords = {name: (pts[idx].x(), pts[idx].y()) for name, idx in KP.items()}

            with STATE.lock:
                STATE.last_pts = coords
                # testa cruzamento de braços
                if not STATE.recording and crossed_above_head(coords):
                    STATE.recording = True
                    pulse_led(3)          # acende LED agora
                    threading.Thread(target=save_clip, daemon=True).start()

    return Gst.PadProbeReturn.OK

# ————————————————————————————————————————————
# Salvamento assíncrono do vídeo
# ————————————————————————————————————————————
def save_clip():
    """
    Copia os últimos 20 s do FRAME_BUFFER, codifica em H264 (MP4) sem
    bloquear o callback de inferência.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"bracos_cruzados_{ts}.mp4")

    # snapshot do buffer para não conflitar com o deque em uso
    with STATE.lock:
        frames = list(FRAME_BUFFER)
    if not frames:
        STATE.recording = False
        return

    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")   # H.264
    out = cv2.VideoWriter(path, fourcc, FPS_ESTIMADO, (w, h))

    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"[SALVO] {path} ({len(frames)/FPS_ESTIMADO:.1f}s)")

    # limpeza / cooldown (pode ajustar se quiser ignorar múltiplos eventos)
    time.sleep(3)      # evita gravar 2× instantaneamente
    STATE.recording = False

# ————————————————————————————————————————————
# Main – inicializa GStreamer e roda
# ————————————————————————————————————————————
def main():
    Gst.init(None)

    user_data = UserCallback()
    app = GStreamerPoseEstimationApp(
        app_callback=app_callback,
        user_data=user_data,
        # por padrão o exemplo da Hailo já pega a Picamera2
        hef_path="yolov8s_pose.hef",
        video_source="rpi"         # usa Picamera2 via libcamera-gst
        # (adicione --video-fps/--video-width/height se quiser ajustar)
    )
    try:
        app.run()                  # loop principal bloqueante
    except KeyboardInterrupt:
        print("Encerrando…")

if __name__ == "__main__":
    main()
