# -*- coding: utf-8 -*-
"""
Script Principal (versão Raspberry Pi 5):
Detecção de braços cruzados com YOLOv8 + Picamera2/Webcam
Gravação de vídeo em buffer circular e acionamento de LED no GPIO 17 (3 s) a
cada detecção confirmada.

Principais melhorias para o RPi 5 + kit de IA
─────────────────────────────────────────────
• Modelo maior (yolov8s‑pose) e imagem de entrada 640×480.
• Taxa de detecção ↑ para 5 Hz (detection_interval = 0.2 s).
• Ajustes de paralelismo : torch.set_num_threads(4) e OMP_NUM_THREADS.
• Otimizações OpenCV (cv2.setUseOptimized(True)).
• Suporte a GPU/NPU (Torch.compile True se CUDA/Metal/TPU disponível).
• LED no GPIO 17 aceso por 3 s quando braços cruzados confirmados.
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import threading
from collections import deque
from datetime import datetime

# ───────────────── HARDWARE/DESEMPENHO ────────────────── #
# Ajuste de threads do Torch/OpenMP para o quad‑core do RPi 5
os.environ.setdefault("OMP_NUM_THREADS", "4")  # força 4 threads
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
cv2.setUseOptimized(True)

# Tentativa opcional de compilar o modelo para GPU/NPU, se disponível
TORCH_COMPILE = False  # mude para True se o backend da sua distro suportar

# ───────────────── GPIO (LED) ──────────────────────────── #
try:
    import RPi.GPIO as GPIO

    LED_AVAILABLE = True
    LED_PIN = 17
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
except (ImportError, RuntimeError):
    LED_AVAILABLE = False
    LED_PIN = None
    print("RPi.GPIO não disponível ‑ LED desativado.")


# Função que acende o LED por "duration" segundos
_last_led_pulse = 0.0  # evita múltiplos disparos simultâneos


def pulse_led(duration: float = 3.0):
    global _last_led_pulse
    if not LED_AVAILABLE:
        return
    now = time.time()
    if now - _last_led_pulse < 0.2:  # já piscando
        return
    _last_led_pulse = now
    GPIO.output(LED_PIN, GPIO.HIGH)

    def _off():
        GPIO.output(LED_PIN, GPIO.LOW)

    threading.Timer(duration, _off, daemon=True).start()


# ─────────── CONFIGURAÇÕES DE GRAVAÇÃO/BUFFER ─────────── #
buffer_tamanho_segundos = 50  # quanto manter no buffer
pasta_gravacao = "gravacoes"
prefixo_arquivo = "braco_cruzado_"
prefixo_manual = "manual_"

# Cria pasta se não existir
os.makedirs(pasta_gravacao, exist_ok=True)

# ───────────── CÂMERAS (PiCam2 ou Webcam) ─────────────── #
usar_picamera = True
webcam_id = 0
webcam_width, webcam_height = 1280, 720
picam_width, picam_height = 1280, 720

# ───────────────── DETECÇÃO ────────────────────────────── #
print("Carregando modelo YOLOv8...")
model = YOLO("yolov8s-pose.pt")  # modelo maior, mas cabe no RPi 5
if TORCH_COMPILE and torch.cuda.is_available():
    model.model = torch.compile(model.model)  # aceleração extra

# Keypoints COCO‑pose
LEFT_WRIST, RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER = 9, 10, 5, 6
LEFT_ELBOW, RIGHT_ELBOW, NOSE = 7, 8, 0

# Taxa de detecção
DETECTION_INTERVAL = 0.2  # 5×/s
DETECTION_SIZE = (640, 480)

# ──────────────── VARIÁVEIS GLOBAIS ───────────────────── #
frame_count = 0
fps_medio = 30

# Flags de estado
executando = True
gravando_video = False

buffer_frames: deque[np.ndarray]
buffer_timestamps: deque[float]

buffer_frames = deque(maxlen=int(fps_medio * buffer_tamanho_segundos))
buffer_timestamps = deque(maxlen=buffer_frames.maxlen)

# Detecção de braços cruzados
bracos_cruzados_start_time = None
BRACOS_CROSSED_THRESHOLD = 0.8
ultimo_estado_gravacao = False
contador_bracos_cruzados = 0
contador_gravacoes_manuais = 0

gravar_flag = False  # marca para gravar automaticamente

# Locks
buffer_lock = threading.Lock()
camera_lock = threading.Lock()

# Threads (serão inicializadas depois)
detection_thread_handle: threading.Thread | None = None
video_writer_thread_handle: threading.Thread | None = None

# Capture devices
picam2 = None
webcam = None

# ───────────── FUNÇÕES AUXILIARES ─────────────────────── #

def inicializar_picamera():
    global picam2
    from picamera2 import Picamera2
    if picam2 is not None:
        try:
            picam2.stop(); picam2.close()
        except Exception:
            pass
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (picam_width, picam_height)},
        controls={"FrameRate": 30},
    )
    picam2.configure(cfg)
    picam2.start()
    print("PiCamera2 iniciada @", picam_width, "×", picam_height)


def inicializar_webcam():
    global webcam
    if webcam is not None:
        webcam.release()
    webcam = cv2.VideoCapture(webcam_id)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
    if not webcam.isOpened():
        raise RuntimeError("Não foi possível abrir a webcam")
    print("Webcam iniciada @", webcam_width, "×", webcam_height)


def capturar_frame():
    with camera_lock:
        if usar_picamera:
            try:
                return picam2.capture_array()
            except Exception as e:
                print("Erro PiCamera:", e)
                return None
        else:
            ret, frm = webcam.read()
            return frm if ret else None

# ─────────────── GRAVAÇÃO DE VÍDEO ────────────────────── #

def thread_gravacao_video(frames, nome, fps, h, w):
    global gravando_video
    try:
        os.nice(-10)
    except Exception:
        pass
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw = cv2.VideoWriter(nome, fourcc, fps, (w, h))
    for f in frames:
        if f is not None:
            vw.write(f)
    vw.release()
    print("Vídeo salvo:", nome)
    gravando_video = False


def salvar_buffer(trigger_frame=None, manual=False):
    global gravando_video, fps_medio, buffer_frames
    if gravando_video:
        return
    gravando_video = True
    tm = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = prefixo_manual if manual else prefixo_arquivo
    nome = os.path.join(pasta_gravacao, f"{prefix}{tm}.avi")

    with buffer_lock:
        frames = list(buffer_frames)
    if trigger_frame is not None:
        frames.append(trigger_frame.copy())
    if not frames:
        gravando_video = False
        return

    h, w = frames[0].shape[:2]
    fps_out = max(10.0, min(fps_medio, 30.0))
    threading.Thread(target=thread_gravacao_video, args=(frames, nome, fps_out, h, w), daemon=True).start()

# ───────────────── DETECÇÃO (THREAD) ───────────────────── #

def detection_loop():
    global bracos_cruzados_start_time, gravar_flag, ultimo_estado_gravacao, contador_bracos_cruzados
    last_det = 0.0
    while executando:
        if time.time() - last_det < DETECTION_INTERVAL:
            time.sleep(0.01)
            continue
        frame = frame_para_detectar  # cópia de referência global
        if frame is None:
            continue
        last_det = time.time()
        fr_resized = cv2.resize(frame, DETECTION_SIZE)
        try:
            results = model(fr_resized, verbose=False)
        except Exception as e:
            print("YOLO erro:", e)
            continue
        algum_braco = False
        if results and hasattr(results[0], "keypoints"):
            scale_x = frame.shape[1] / DETECTION_SIZE[0]
            scale_y = frame.shape[0] / DETECTION_SIZE[1]
            for kp in results[0].keypoints.data.cpu().numpy():
                # salva confiança
                if kp.shape[0] < RIGHT_WRIST + 1:
                    continue
                lw, rw = kp[LEFT_WRIST][:2], kp[RIGHT_WRIST][:2]
                ls, rs = kp[LEFT_SHOULDER][:2], kp[RIGHT_SHOULDER][:2]
                lw *= (scale_x, scale_y); rw *= (scale_x, scale_y)
                ls *= (scale_x, scale_y); rs *= (scale_x, scale_y)
                wrists_crossed = lw[0] > rs[0] and rw[0] < ls[0] and lw[0] > rw[0]
                close_horiz = abs(lw[0] - rw[0]) < frame.shape[1] * 0.3
                arms_above_head = lw[1] < kp[NOSE][1] * scale_y and rw[1] < kp[NOSE][1] * scale_y
                if wrists_crossed and close_horiz and arms_above_head:
                    algum_braco = True
                    break
        now = time.time()
        if algum_braco:
            if bracos_cruzados_start_time is None:
                bracos_cruzados_start_time = now
            elif now - bracos_cruzados_start_time >= BRACOS_CROSSED_THRESHOLD:
                gravar_flag = True
                if not ultimo_estado_gravacao:
                    contador_bracos_cruzados += 1
                    pulse_led(3.0)  # aciona LED
                ultimo_estado_gravacao = True
        else:
            bracos_cruzados_start_time = None
            gravar_flag = False
            ultimo_estado_gravacao = False

# ───────────────── COORDENAÇÃO DE GRAVAÇÃO ─────────────── #

def grava_loop():
    last_video = 0.0
    MIN_INTERVAL = 5.0
    while executando:
        if gravar_flag and not gravando_video and time.time() - last_video > MIN_INTERVAL:
            salvar_buffer(frame_atual, manual=False)
            last_video = time.time()
        time.sleep(0.1)

# ───────────────── INICIALIZAÇÃO ───────────────────────── #
if usar_picamera:
    inicializar_picamera()
else:
    inicializar_webcam()

print("▶ Iniciado (RPi 5) – pressione 'q' para sair…")

# thread detection & grava
frame_para_detectar = None
frame_atual = None

detection_thread_handle = threading.Thread(target=detection_loop, daemon=True)
detection_thread_handle.start()
video_writer_thread_handle = threading.Thread(target=grava_loop, daemon=True)
video_writer_thread_handle.start()

# ───────────────── LOOP PRINCIPAL ──────────────────────── #
frames_ok = 0
ultimo_frame_time = time.time()

while True:
    frame_rgb = capturar_frame()
    if frame_rgb is None:
        print("Frame nulo – reconectando câmera…")
        time.sleep(0.1)
        continue
    frame = cv2.flip(frame_rgb, 1)
    timestamp = time.time()

    # Buffer
    with buffer_lock:
        buffer_frames.append(frame.copy())
        buffer_timestamps.append(timestamp)

    # Atualiza FPS estimado
    frames_ok += 1
    if timestamp - ultimo_frame_time >= 1.0:
        fps_medio = frames_ok / (timestamp - ultimo_frame_time)
        frames_ok = 0
        ultimo_frame_time = timestamp
        # redimensiona buffer
        novo_max = int(fps_medio * buffer_tamanho_segundos)
        if novo_max != buffer_frames.maxlen:
            with buffer_lock:
                buffer_frames = deque(buffer_frames, maxlen=novo_max)
                buffer_timestamps = deque(buffer_timestamps, maxlen=novo_max)
            print(f"Buffer → {novo_max} frames @ {fps_medio:.1f} fps")

    # torna disponível para detecção
    frame_atual = frame
    frame_para_detectar = frame

    # Exibição simples (desconto gráfico pesado)
    cv2.putText(frame, f"FPS {fps_medio:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Auto vídeos: {contador_bracos_cruzados}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"LED GPIO17 {'ON' if LED_AVAILABLE and GPIO.input(LED_PIN) else 'OFF'}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

    cv2.imshow("Detecção de Braços Cruzados", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('g'):
        salvar_buffer(frame.copy(), manual=True); contador_gravacoes_manuais += 1

# ───────────────── FINALIZAÇÃO ─────────────────────────── #
executando = False
if detection_thread_handle:
    detection_thread_handle.join(1.0)
if video_writer_thread_handle:
    video_writer_thread_handle.join(1.0)

if LED_AVAILABLE:
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.cleanup()

if picam2:
    picam2.stop(); picam2.close()
if webcam:
    webcam.release()
cv2.destroyAllWindows()

print("Programa encerrado. Vida longa ao RPi 5! 😊")
