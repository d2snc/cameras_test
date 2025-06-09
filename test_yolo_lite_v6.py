# -*- coding: utf-8 -*-
"""
Script Principal: Captura, detecção e gravação com YOLOv8 + PiCamera2/Webcam,
com indicação luminosa (LED no GPIO-17) usando libgpiod.

Requisitos:
  sudo apt-get install python3-libgpiod python3-opencv python3-numpy
  pip install ultralytics gpiod
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import threading
import os
from collections import deque
from datetime import datetime

# ───────── CONFIGURAÇÕES GERAIS ───────── #
buffer_tamanho_segundos = 50        # Quantos segundos manter no buffer circular
pasta_gravacao = "gravacoes"
prefixo_arquivo = "braco_cruzado_"
prefixo_manual = "manual_"

# Câmeras
usar_picamera = True                # True → PiCamera2, False → webcam
webcam_id = 0
webcam_width, webcam_height = 1280, 720
picam_width,  picam_height  = 1280, 720

# ───────── GPIO via libgpiod ───────── #
try:
    import gpiod

    CHIP_NAME = "gpiochip0"         # /dev/gpiochip0
    LED_LINE_OFFSET = 17            # GPIO-17 (BCM) = pino físico 11
    chip = gpiod.Chip(CHIP_NAME)
    led_line = chip.get_line(LED_LINE_OFFSET)
    led_line.request(
        consumer="yolo-led",
        type=gpiod.LINE_REQ_DIR_OUT,
        default_vals=[0],
    )
    LED_AVAILABLE = True
except Exception as e:
    print(f"LED desativado (libgpiod indisponível): {e}")
    LED_AVAILABLE = False

_last_led_pulse = 0.0  # evita pulsos sobrepostos


def pulse_led(duration: float = 3.0):
    """Acende o LED por 'duration' s, ignorando pulsos muito próximos."""
    global _last_led_pulse
    if not LED_AVAILABLE:
        return
    now = time.time()
    if now - _last_led_pulse < 0.2:      # já piscando recentemente
        return
    _last_led_pulse = now

    led_line.set_value(1)                # liga LED

    def _off():
        led_line.set_value(0)            # apaga LED

    t = threading.Timer(duration, _off)
    t.daemon = True
    t.start()

# ───────── MODELO YOLO ───────── #
print("Carregando modelo YOLOv8...")
model = YOLO("yolov8n-pose.pt")  # modelo leve

# Índices de keypoints (COCO-pose)
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
NOSE = 0

class DetectionResult:
    def __init__(self):
        self.pose_data = []
        self.braco_cruzado = False
        self.num_pessoas = 0
        self.duracao_bracos_cruzados = 0

# ───────── VARIÁVEIS GLOBAIS ───────── #
frame_count = 0
grava = False
gravando_video = False
detection_interval = 0.2               # 5 detecções por segundo
bracos_cruzados_start_time = None
bracos_cruzados_threshold = 0.8        # 0,8 s para considerar braço cruzado
contador_bracos_cruzados = 0
contador_gravacoes_manuais = 0
ultimo_estado_gravacao = False
buffer_frames = deque()                # será configurado após medir FPS
buffer_timestamps = deque()
modo_visualizacao_completa = True
mostrar_diagnostico = True
fps_medio = 30

detection_result = DetectionResult()
frame_para_detectar = None
frame_atual = None
executando = True
buffer_lock = threading.Lock()
camera_lock = threading.Lock()

picam2 = None
webcam = None

# ───────── FUNÇÕES DE CÂMERA ───────── #
def inicializar_picamera():
    global picam2
    from picamera2 import Picamera2
    if picam2:
        try:
            picam2.stop(); picam2.close()
        except Exception:
            pass
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (picam_width, picam_height)}
    )
    picam2.configure(cfg)
    picam2.start()
    print("PiCamera2 inicializada:", picam_width, "x", picam_height)

def inicializar_webcam():
    global webcam
    if webcam:
        webcam.release()
    webcam = cv2.VideoCapture(webcam_id)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH,  webcam_width)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
    if not webcam.isOpened():
        raise RuntimeError("Erro ao abrir webcam")
    print("Webcam inicializada:", webcam_width, "x", webcam_height)

def alternar_camera():
    global usar_picamera
    with camera_lock:
        usar_picamera = not usar_picamera
        if usar_picamera:
            if webcam: webcam.release()
            inicializar_picamera()
        else:
            if picam2:
                try: picam2.stop(); picam2.close()
                except Exception: pass
            inicializar_webcam()
        print("Agora usando", "PiCamera" if usar_picamera else "Webcam")

def capturar_frame():
    with camera_lock:
        if usar_picamera:
            try: return picam2.capture_array()
            except Exception as e:
                print("Erro PiCamera:", e); return None
        ret, frame = webcam.read()
        if not ret:
            print("Erro webcam"); return None
        return frame

# ───────── GRAVAÇÃO DE VÍDEO ───────── #
def thread_gravacao_video_prioridade(frames, nome, fps, h, w, manual=False):
    global gravando_video
    try: os.nice(-10)
    except Exception: pass

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vw = cv2.VideoWriter(nome, fourcc, fps, (w, h))
    if not vw.isOpened():
        print("Falha ao abrir VideoWriter"); gravando_video = False; return

    for f in frames:
        vw.write(f)
    vw.release()
    print(f"Vídeo {'manual' if manual else 'automático'} salvo:", nome)
    gravando_video = False

def salvar_buffer_como_video(trigger_frame=None, manual=False):
    global gravando_video, contador_gravacoes_manuais
    if gravando_video:
        print("Gravação já em andamento"); return
    gravando_video = True
    if manual: contador_gravacoes_manuais += 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = prefixo_manual if manual else prefixo_arquivo
    nome = os.path.join(pasta_gravacao, f"{prefix}{ts}.avi")

    with buffer_lock:
        frames = [f.copy() for f in buffer_frames] + (
            [trigger_frame.copy()] if trigger_frame is not None else []
        )
    if not frames:
        print("Buffer vazio"); gravando_video = False; return

    h, w = frames[0].shape[:2]
    fps = max(10.0, min(fps_medio, 30.0))
    threading.Thread(
        target=thread_gravacao_video_prioridade,
        args=(frames, nome, fps, h, w, manual),
        daemon=True
    ).start()

def video_writer_thread():
    global gravando_video, grava, ultimo_estado_gravacao
    ultimo_time = 0
    while executando:
        if grava and not ultimo_estado_gravacao:
            t = time.time()
            if t - ultimo_time >= 5:
                salvar_buffer_como_video(frame_atual, manual=False)
                ultimo_time = t
        ultimo_estado_gravacao = grava
        time.sleep(0.1)

# ───────── DETECÇÃO ───────── #
def detection_thread():
    global detection_result, bracos_cruzados_start_time, grava, ultimo_estado_gravacao
    det_w, det_h = 480, 360
    last_det = 0
    while executando:
        if frame_para_detectar is not None and time.time() - last_det >= detection_interval:
            last_det = time.time()
            try:
                res = model(cv2.resize(frame_para_detectar, (det_w, det_h)), verbose=False)
            except Exception as e:
                print("YOLO erro:", e); continue
            algum = False; pessoas = 0; detalhes = []
            if res and hasattr(res[0], "keypoints") and res[0].keypoints.data.any():
                scale_x = frame_para_detectar.shape[1] / det_w
                scale_y = frame_para_detectar.shape[0] / det_h
                for kp in res[0].keypoints.data:
                    kp = kp.cpu().numpy()
                    pessoas += 1
                    if kp.shape[0] < RIGHT_WRIST+1: continue
                    lw, rw = kp[LEFT_WRIST], kp[RIGHT_WRIST]
                    ls, rs = kp[LEFT_SHOULDER], kp[RIGHT_SHOULDER]
                    le, re = kp[LEFT_ELBOW], kp[RIGHT_ELBOW]
                    nose   = kp[NOSE]

                    confs = [lw[2], rw[2], ls[2], rs[2], le[2], re[2], nose[2]]
                    if min(confs) < 0.5:
                        detalhes.append({'braco_cruzado': False}); continue

                    lw_xy = (lw[:2] * [scale_x, scale_y]).astype(int)
                    rw_xy = (rw[:2] * [scale_x, scale_y]).astype(int)
                    ls_xy = (ls[:2] * [scale_x, scale_y]).astype(int)
                    rs_xy = (rs[:2] * [scale_x, scale_y]).astype(int)
                    le_xy = (le[:2] * [scale_x, scale_y]).astype(int)
                    re_xy = (re[:2] * [scale_x, scale_y]).astype(int)
                    ns_xy = (nose[:2]* [scale_x, scale_y]).astype(int)

                    cruz = (lw_xy[0] > rs_xy[0] and rw_xy[0] < ls_xy[0] and lw_xy[0] > rw_xy[0])
                    acima = (lw_xy[1] < ns_xy[1] and rw_xy[1] < ns_xy[1])
                    prox  = abs(lw_xy[0]-rw_xy[0]) < frame_para_detectar.shape[1]*0.3
                    pessoa_ok = cruz and acima and prox
                    if pessoa_ok: algum = True

                    detalhes.append({
                        'braco_cruzado': pessoa_ok,
                        'coords': dict(
                            left_wrist=tuple(lw_xy), right_wrist=tuple(rw_xy),
                            left_shoulder=tuple(ls_xy), right_shoulder=tuple(rs_xy),
                            left_elbow=tuple(le_xy), right_elbow=tuple(re_xy),
                            nose=tuple(ns_xy)
                        ),
                        'confidences': dict(zip(
                            ['lw','rw','ls','rs','le','re','nose'],
                            [c>0.5 for c in confs]
                        )),
                        'debug_info': dict(cruzados=cruz, acima_cabeca=acima, proximos=prox)
                    })

            # Atualiza estado de gravação
            t = time.time()
            if algum:
                if bracos_cruzados_start_time is None:
                    bracos_cruzados_start_time = t
                elif t - bracos_cruzados_start_time >= bracos_cruzados_threshold:
                    grava = True
                    if not ultimo_estado_gravacao:
                        contador_bracos_cruzados += 1
                        pulse_led(3)
                        ultimo_estado_gravacao = True
            else:
                bracos_cruzados_start_time = None
                grava = False
                ultimo_estado_gravacao = False

            detection_result.pose_data = detalhes
            detection_result.braco_cruzado = algum
            detection_result.num_pessoas = pessoas
            detection_result.duracao_bracos_cruzados = (
                0 if bracos_cruzados_start_time is None
                else t - bracos_cruzados_start_time
            )
        time.sleep(0.01)

# ───────── MAIN ───────── #
os.makedirs(pasta_gravacao, exist_ok=True)
if usar_picamera: inicializar_picamera()
else: inicializar_webcam()

detection_th = threading.Thread(target=detection_thread, daemon=True).start()
video_th = threading.Thread(target=video_writer_thread, daemon=True).start()

print("Pressione 'q' para sair | 'g' grava manual | '+'/'-' ajusta buffer | 'v' visualização | 'd' diag | 'c' troca câmera")

ultimo_frame_time = time.time()
tamanho_buffer = int(fps_medio * buffer_tamanho_segundos)
buffer_frames = deque(maxlen=tamanho_buffer)
buffer_timestamps = deque(maxlen=tamanho_buffer)
frames_ok = 0

while True:
    frame_rgb = capturar_frame()
    if frame_rgb is None:
        time.sleep(0.1); continue

    frame = cv2.flip(frame_rgb, 1)
    ts = time.time()
    with buffer_lock:
        buffer_frames.append(frame.copy())
        buffer_timestamps.append(ts)

    frames_ok += 1
    if ts - ultimo_frame_time >= 1:
        fps_medio = frames_ok / (ts - ultimo_frame_time)
        novo_len = int(fps_medio * buffer_tamanho_segundos)
        if abs(novo_len - buffer_frames.maxlen) > fps_medio * 0.1:
            with buffer_lock:
                buffer_frames = deque(buffer_frames, maxlen=novo_len)
                buffer_timestamps = deque(buffer_timestamps, maxlen=novo_len)
            print(f"Buffer ajustado: {novo_len} frames ({buffer_tamanho_segundos}s)")
        ultimo_frame_time, frames_ok = ts, 0

    frame_atual = frame
    frame_para_detectar = frame

    # Desenho/overlay
    if modo_visualizacao_completa:
        for pdata in detection_result.pose_data:
            if 'coords' not in pdata: continue
            c = pdata['coords']; vis = pdata['confidences']
            if vis['lw']: cv2.circle(frame, c['left_wrist'], 8, (255,0,0), -1)
            if vis['rw']: cv2.circle(frame, c['right_wrist'],8,(0,255,0), -1)
            if vis['ls']: cv2.circle(frame, c['left_shoulder'],8,(0,0,255), -1)
            if vis['rs']: cv2.circle(frame, c['right_shoulder'],8,(255,255,0), -1)
            if vis['le']: cv2.circle(frame, c['left_elbow'],8,(0,255,255), -1)
            if vis['re']: cv2.circle(frame, c['right_elbow'],8,(255,0,255), -1)
            if vis['nose']: cv2.circle(frame, c['nose'],8,(255,255,255), -1)
            status = "Braços cruzados" if pdata['braco_cruzado'] else "Braços abertos"
            cor = (0,255,0) if pdata['braco_cruzado'] else (0,0,255)
            cv2.putText(frame, status, (c['nose'][0]-80, c['nose'][1]-30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,cor,2)
    else:
        for i,pdata in enumerate(detection_result.pose_data):
            cv2.putText(frame,"X" if pdata['braco_cruzado'] else "O",
                        (20+i*30, 50), cv2.FONT_HERSHEY_SIMPLEX,1,
                        (0,0,255) if pdata['braco_cruzado'] else (0,255,0),2)

    cv2.putText(frame, f"Grava: {grava}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0) if grava else (0,0,255),2)
    cv2.putText(frame, f"Pessoas: {detection_result.num_pessoas}", (10,70),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(frame, f"Detec auto: {contador_bracos_cruzados}", (10,110),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)
    cv2.putText(frame, f"Buffer: {len(buffer_frames)}/{buffer_frames.maxlen}",
                (10,150), cv2.FONT_HERSHEY_SIMPLEX,0.7,(150,150,150),2)
    cv2.putText(frame, f"FPS: {fps_medio:.1f}", (10,180),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(150,150,150),2)
    cv2.putText(frame,"g:Gravar +/-:Buffer v:Vis d:Diag c:Câmera q:Sair",
                (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),2)

    if gravando_video:
        cv2.putText(frame,"SALVANDO...",(frame.shape[1]//2-100,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)

    cv2.imshow("Detecção Braços (YOLOv8)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key in (ord('+'), ord('=')):
        buffer_tamanho_segundos += 5
    elif key in (ord('-'), ord('_')) and buffer_tamanho_segundos > 5:
        buffer_tamanho_segundos -= 5
    elif key == ord('g'):
        threading.Thread(target=salvar_buffer_como_video,
                         args=(frame.copy(), True), daemon=True).start()
    elif key == ord('v'):
        modo_visualizacao_completa = not modo_visualizacao_completa
    elif key == ord('d'):
        mostrar_diagnostico = not mostrar_diagnostico
    elif key == ord('c'):
        alternar_camera()

# ───────── ENCERRAMENTO ───────── #
executando = False
if LED_AVAILABLE:
    try:
        led_line.set_value(0); led_line.release(); chip.close()
    except Exception: pass
if picam2:
    try: picam2.stop(); picam2.close()
    except Exception: pass
if webcam: webcam.release()
cv2.destroyAllWindows()
print("Programa finalizado.")
