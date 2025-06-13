# -*- coding: utf-8 -*-
"""
Script Principal: Captura, detecção e gravação com Hailo + GStreamerPoseEstimationApp.
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import cv2
import numpy as np
import threading
import time
from collections import deque
from datetime import datetime
import hailo
from hailo_apps_infra.hailo_rpi_common import get_caps_from_pad, get_numpy_from_buffer, app_callback_class
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# Inicializa GStreamer
Gst.init(None)

# -- CONFIGURAÇÕES DE GRAVAÇÃO/DETALHES --
buffer_tamanho_segundos = 50  # Quantos segundos manter no buffer
pasta_gravacao = "gravacoes"   # Pasta onde os vídeos serão salvos
prefixo_arquivo = "braco_cruzado_"
prefixo_manual = "manual_"

# Cria pasta de gravação se não existir
if not os.path.exists(pasta_gravacao):
    os.makedirs(pasta_gravacao)

# -- VARIÁVEIS GLOBAIS --
frame_atual = None
frame_para_detectar = None
grava = False
gravando_video = False
bracos_cruzados_start_time = None
bracos_cruzados_threshold = 0.8  # segundos
contador_bracos_cruzados = 0
contador_gravacoes_manuais = 0
ultimo_estado_gravacao = False

tamanho_buffer = 1
buffer_frames = deque(maxlen=tamanho_buffer)
buffer_timestamps = deque(maxlen=tamanho_buffer)
fps_medio = 30
buffer_lock = threading.Lock()

# Índices de keypoints COCO
KEYPOINTS = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,'left_hip': 11,'right_hip': 12,
    'left_knee': 13,'right_knee': 14,'left_ankle': 15,'right_ankle': 16
}

# Classe callback Hailo\class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()

# Função de callback do pipeline
def app_callback(pad, info, user_data):
    global frame_atual, frame_para_detectar, grava, bracos_cruzados_start_time, último_estado_gravacao, contador_bracos_cruzados
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Extrai frame se configurado
    fmt, width, height = get_caps_from_pad(pad)
    frame = None
    if user_data.use_frame and fmt and width and height:
        frame = get_numpy_from_buffer(buffer, fmt, width, height)
        # converte para BGR para OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_atual = frame.copy()
        frame_para_detectar = frame.copy()

    # Extrai detecções do buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Processa detecções COCO-pose
    algum_braco_cruzado = False
    for det in detections:
        label = det.get_label()
        if label != 'person':
            continue
        bbox = det.get_bbox()
        landmarks = det.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks:
            continue
        points = landmarks[0].get_points()
        # converte keypoints
        coords = {k: (
            int((points[i].x() * bbox.width() + bbox.xmin()) * width),
            int((points[i].y() * bbox.height() + bbox.ymin()) * height)
        ) for k, i in KEYPOINTS.items()}
        # critérios de braços cruzados
        lwx, lwy = coords['left_wrist']; rwx, rwy = coords['right_wrist']
        lsx, lsy = coords['left_shoulder']; rsx, rsy = coords['right_shoulder']
        nx, ny = coords['nose']
        wrists_crossed = (lwx > rsx and rwx < lsx and lwx > rwx)
        arms_above_head = (lwy < ny and rwy < ny)
        close_horiz = abs(lwx - rwx) < width * 0.3
        if wrists_crossed and arms_above_head and close_horiz:
            algum_braco_cruzado = True
            break

    # Lógica de gravação automática
    now = time.time()
    if algum_braco_cruzado:
        if bracos_cruzados_start_time is None:
            bracos_cruzados_start_time = now
        elif now - bracos_cruzados_start_time >= bracos_cruzados_threshold:
            grava = True
            if not ultimo_estado_gravacao:
                contador_bracos_cruzados += 1
                ultimo_estado_gravacao = True
    else:
        bracos_cruzados_start_time = None
        grava = False
        ultimo_estado_gravacao = False

    return Gst.PadProbeReturn.OK

# Thread de gravação de vídeo
def video_writer_thread():
    global gravando_video
    while True:
        if grava and not gravando_video:
            with buffer_lock:
                frames = list(buffer_frames)
            if frames:
                gravando_video = True
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                nome = os.path.join(pasta_gravacao, f"{prefixo_arquivo}{timestamp}.avi")
                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(nome, fourcc, min(fps_medio, 30), (width, height))
                for f in frames:
                    writer.write(f)
                writer.release()
                gravando_video = False
        time.sleep(0.1)

if __name__ == '__main__':
    # Inicializa buffer de frames
    tamanho_buffer = int(fps_medio * buffer_tamanho_segundos)
    buffer_frames = deque(maxlen=tamanho_buffer)
    buffer_timestamps = deque(maxlen=tamanho_buffer)

    # Inicia thread de gravação
    threading.Thread(target=video_writer_thread, daemon=True).start()

    # Cria e executa o app Hailo
    user_data = user_app_callback_class()
    user_data.use_frame = True
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    try:
        app.run()  # Loop principal GStreamer
    except KeyboardInterrupt:
        pass

    # Finaliza GStreamer e libera recursos
    app.stop()
    cv2.destroyAllWindows()
