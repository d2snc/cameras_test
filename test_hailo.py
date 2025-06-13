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

fps_medio = 30  # estimativa inicial
buffer_lock = threading.Lock()

tamanho_buffer = int(fps_medio * buffer_tamanho_segundos)
buffer_frames = deque(maxlen=tamanho_buffer)
buffer_timestamps = deque(maxlen=tamanho_buffer)

# Índices de keypoints COCO
KEYPOINTS = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16
}

# -----------------------------------------------------------------------------------------------
# Classe de callback para usar no pipeline Hailo
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()

# -----------------------------------------------------------------------------------------------
# Função de callback chamada para cada buffer de vídeo
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    global frame_atual, frame_para_detectar, grava
    global bracos_cruzados_start_time, ultimo_estado_gravacao, contador_bracos_cruzados

    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Extrai frame se configurado
    fmt, width, height = get_caps_from_pad(pad)
    if user_data.use_frame and fmt and width and height:
        frame = get_numpy_from_buffer(buffer, fmt, width, height)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_atual = frame.copy()
        frame_para_detectar = frame.copy()

    # Extrai detecções do buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    algum_braco_cruzado = False
    # Verifica cada detecção de pessoa
    for det in detections:
        if det.get_label() != 'person':
            continue
        landmarks = det.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks:
            continue
        bbox = det.get_bbox()
        points = landmarks[0].get_points()
        coords = {}
        for name, idx in KEYPOINTS.items():
            pt = points[idx]
            x = int((pt.x() * bbox.width() + bbox.xmin()) * width)
            y = int((pt.y() * bbox.height() + bbox.ymin()) * height)
            coords[name] = (x, y)

        # Critérios de braços cruzados
        lwx, lwy = coords['left_wrist']
        rwx, rwy = coords['right_wrist']
        lsx, lsy = coords['left_shoulder']
        rsx, rsy = coords['right_shoulder']
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

# Thread que grava vídeos a partir do buffer
class VideoWriterThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)

    def run(self):
        global gravando_video
        while True:
            if grava and not gravando_video:
                with buffer_lock:
                    frames = list(buffer_frames)
                if frames:
                    gravando_video = True
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = os.path.join(pasta_gravacao, f"{prefixo_arquivo}{ts}.avi")
                    h, w = frames[0].shape[:2]
                    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'),
                                             min(fps_medio, 30), (w, h))
                    for f in frames:
                        writer.write(f)
                    writer.release()
                    gravando_video = False
            time.sleep(0.1)

if __name__ == '__main__':
    # Redimensiona buffer de frames conforme FPS estimado
    tamanho_buffer = int(fps_medio * buffer_tamanho_segundos)
    buffer_frames = deque(maxlen=tamanho_buffer)
    buffer_timestamps = deque(maxlen=tamanho_buffer)

    # Inicia thread de gravação
    VideoWriterThread().start()

    # Cria e executa o aplicativo Hailo
    user_data = user_app_callback_class()
    user_data.use_frame = True
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
        cv2.destroyAllWindows()
