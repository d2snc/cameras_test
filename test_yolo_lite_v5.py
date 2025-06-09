# -*- coding: utf-8 -*-
"""
Script Principal: Captura, detecção e gravação com YOLOv8 + Picamera2/Webcam.
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

# -- CONFIGURAÇÕES DE GRAVAÇÃO/DETALHES --
buffer_tamanho_segundos = 50  # Quantos segundos manter no buffer
pasta_gravacao = "gravacoes"   # Pasta onde os vídeos serão salvos
prefixo_arquivo = "braco_cruzado_"
prefixo_manual = "manual_"

# -- CONFIGURAÇÃO DE CÂMERA --
usar_picamera = True  # Flag para escolher entre Picamera ou Webcam
webcam_id = 0         # ID da webcam (normalmente 0 para a primeira webcam)
webcam_width = 1280   # Largura para webcam
webcam_height = 720   # Altura para webcam
picam_width = 1280    # Largura para PiCamera 
picam_height = 720    # Altura para PiCamera

# Cria pasta de gravação se não existir
if not os.path.exists(pasta_gravacao):
    os.makedirs(pasta_gravacao)

# -- VARIÁVEIS GLOBAIS --
frame_count = 0
grava = False               # Indica se deve gravar automaticamente (braços cruzados)
gravando_video = False      # Indica se já está gravando um vídeo (para não gravar em paralelo)
detection_interval = 0.333  # 2 detecções por segundo
bracos_cruzados_start_time = None
bracos_cruzados_threshold = 0.8  # 1 segundo para considerar braços cruzados

contador_bracos_cruzados = 0
contador_gravacoes_manuais = 0
ultimo_estado_gravacao = False

# Variáveis para buffer circular (frames recentes)
buffer_frames = deque(maxlen=0)  # Será inicializado após conhecer o FPS
buffer_timestamps = deque(maxlen=0)  # Timestamps correspondentes

# [OTIMIZAÇÃO 7] Modo de visualização reduzida para menor processamento
modo_visualizacao_completa = True  # Pode ser alterado para False para menos processamento gráfico
mostrar_diagnostico = True         # Mostrar informações de diagnóstico da detecção

# [OTIMIZAÇÃO 2] Modelo YOLOv8 - mantém o mais leve
print("Carregando modelo YOLOv8...")
model = YOLO("yolov8n-pose.pt")  # Modelo mais leve

# Índices de keypoints (modelo COCO-pose)
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
NOSE = 0

# Classe para armazenar resultados de detecção
class DetectionResult:
    def __init__(self):
        self.pose_data = []
        self.braco_cruzado = False
        self.num_pessoas = 0
        self.duracao_bracos_cruzados = 0

detection_result = DetectionResult()
frame_para_detectar = None
frame_atual = None
executando = True
buffer_lock = threading.Lock()
camera_lock = threading.Lock()
fps_medio = 30  # Valor inicial, será atualizado durante a execução

# Variáveis de captura
picam2 = None
webcam = None

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

# -- FUNÇÕES DE GRAVAÇÃO --
def thread_gravacao_video_prioridade(frames_para_gravar, nome_arquivo, fps_gravacao, altura, largura, manual=False):
    global gravando_video, contador_gravacoes_manuais
    try:
        os.nice(-10)
    except Exception as e:
        print("Erro ao alterar prioridade:", e)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(nome_arquivo, fourcc, fps_gravacao, (largura, altura))
    print(f"Usando codec XVID para gravar {len(frames_para_gravar)} frames")
    
    if not video_writer.isOpened():
        print(f"Erro: Não foi possível inicializar o VideoWriter para {nome_arquivo}")
        gravando_video = False
        return False
    
    frame_count_local = 0
    start_time = time.time()
    print(f"Iniciando gravação de {len(frames_para_gravar)} frames...")
    
    for i, frame in enumerate(frames_para_gravar):
        if frame is not None:
            video_writer.write(frame)
            frame_count_local += 1
            if len(frames_para_gravar) > 10 and i % (len(frames_para_gravar) // 10) == 0:
                print(f"Progresso: {i / len(frames_para_gravar) * 100:.1f}% ({i}/{len(frames_para_gravar)} frames)")
    
    video_writer.release()
    tempo_gravacao = time.time() - start_time
    tipo_gravacao = "manual" if manual else "automática"
    print(f"Vídeo {tipo_gravacao} salvo: {nome_arquivo}")
    print(f"Gravados {frame_count_local} frames de {len(frames_para_gravar)} em {tempo_gravacao:.2f} segundos")
    print(f"Taxa de gravação: {frame_count_local/tempo_gravacao:.2f} fps")
    
    gravando_video = False
    return True

def salvar_buffer_como_video(trigger_frame=None, manual=False):
    global buffer_frames, buffer_timestamps, gravando_video, contador_gravacoes_manuais, fps_medio
    
    if gravando_video:
        print("Já existe uma gravação em andamento. Aguarde...")
        return
    
    gravando_video = True
    
    if manual:
        contador_gravacoes_manuais += 1
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = prefixo_manual if manual else prefixo_arquivo
    nome_arquivo = os.path.join(pasta_gravacao, f"{prefix}{timestamp}.avi")
    
    print(f"Tamanho do buffer antes da cópia: {len(buffer_frames)} frames")
    
    frames_para_gravar = []
    with buffer_lock:
        if len(buffer_frames) == 0:
            print("ERRO: Buffer vazio, nada para gravar")
            gravando_video = False
            return
        for frame in buffer_frames:
            frames_para_gravar.append(frame.copy())
        if trigger_frame is not None:
            frames_para_gravar.append(trigger_frame.copy())
    
    print(f"Frames para gravar após cópia do buffer: {len(frames_para_gravar)}")
    
    if len(frames_para_gravar) == 0:
        print("ERRO: Nenhum frame copiado para gravação")
        gravando_video = False
        return
    
    altura, largura = frames_para_gravar[0].shape[:2]
    fps_gravacao = max(10.0, min(fps_medio, 30.0))
    
    print(f"Iniciando gravação com {len(frames_para_gravar)} frames a {fps_gravacao:.1f} fps...")
    print(f"Buffer configurado para {buffer_tamanho_segundos} segundos")
    
    threading.Thread(
        target=thread_gravacao_video_prioridade,
        args=(frames_para_gravar, nome_arquivo, fps_gravacao, altura, largura, manual),
        daemon=True
    ).start()
    
    return nome_arquivo

# Thread que coordena gravação automática
def video_writer_thread():
    global executando, grava, ultimo_estado_gravacao
    ultimo_estado_grava = False
    ultimo_arquivo_gravado_time = 0
    min_intervalo_gravacao = 5  # Segundos mínimos entre gravações automáticas
    while executando:
        if grava and not ultimo_estado_grava:
            tempo_atual = time.time()
            if tempo_atual - ultimo_arquivo_gravado_time >= min_intervalo_gravacao:
                with buffer_lock:
                    print(f"Estado do buffer no momento da detecção: {len(buffer_frames)} frames")
                    if len(buffer_frames) > 0:
                        buffer_age = tempo_atual - buffer_timestamps[0]
                        print(f"Buffer contém frames de {buffer_age:.1f} segundos atrás")
                salvar_buffer_como_video(frame_atual, manual=False)
                ultimo_arquivo_gravado_time = tempo_atual
        ultimo_estado_grava = grava
        time.sleep(0.1)

# Thread de detecção com frames reduzidos
def detection_thread():
    global detection_result, frame_para_detectar, executando
    global bracos_cruzados_start_time, contador_bracos_cruzados, ultimo_estado_gravacao, grava
    last_detection_time = 0
    detection_width, detection_height = 480, 360
    while executando:
        try:
            current_time = time.time()
            if current_time - last_detection_time >= detection_interval and frame_para_detectar is not None:
                frame_orig = frame_para_detectar.copy()
                frame_to_process = cv2.resize(frame_orig, (detection_width, detection_height))
                try:
                    results = model(frame_to_process, verbose=False)
                except Exception as e:
                    print(f"Erro ao executar modelo YOLOv8: {e}")
                    results = []
                last_detection_time = current_time
                algum_braco_cruzado = False
                num_pessoas = 0
                detected_people = []
                
                if len(results) > 0 and hasattr(results[0], 'keypoints') and hasattr(results[0].keypoints, 'data') and len(results[0].keypoints.data) > 0:
                    scale_x = frame_orig.shape[1] / detection_width
                    scale_y = frame_orig.shape[0] / detection_height
                    for person in results[0].keypoints.data:
                        try:
                            num_pessoas += 1
                            keypoints = person.cpu().numpy()
                            required_keypoints = max(LEFT_WRIST, RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, NOSE) + 1
                            if keypoints.shape[0] < required_keypoints:
                                print(f"Keypoints insuficientes: encontrados {keypoints.shape[0]}, necessários {required_keypoints}")
                                continue
                            
                            left_wrist = keypoints[LEFT_WRIST][:2] * np.array([scale_x, scale_y])
                            right_wrist = keypoints[RIGHT_WRIST][:2] * np.array([scale_x, scale_y])
                            left_shoulder = keypoints[LEFT_SHOULDER][:2] * np.array([scale_x, scale_y])
                            right_shoulder = keypoints[RIGHT_SHOULDER][:2] * np.array([scale_x, scale_y])
                            left_elbow = keypoints[LEFT_ELBOW][:2] * np.array([scale_x, scale_y])
                            right_elbow = keypoints[RIGHT_ELBOW][:2] * np.array([scale_x, scale_y])
                            nose = keypoints[NOSE][:2] * np.array([scale_x, scale_y])
                            
                            left_wrist_conf = keypoints[LEFT_WRIST][2]
                            right_wrist_conf = keypoints[RIGHT_WRIST][2]
                            left_shoulder_conf = keypoints[LEFT_SHOULDER][2]
                            right_shoulder_conf = keypoints[RIGHT_SHOULDER][2]
                            left_elbow_conf = keypoints[LEFT_ELBOW][2]
                            right_elbow_conf = keypoints[RIGHT_ELBOW][2]
                            nose_conf = keypoints[NOSE][2]
                            
                            lw = (int(left_wrist[0]), int(left_wrist[1]))
                            rw = (int(right_wrist[0]), int(right_wrist[1]))
                            ls = (int(left_shoulder[0]), int(left_shoulder[1]))
                            rs = (int(right_shoulder[0]), int(right_shoulder[1]))
                            le = (int(left_elbow[0]), int(left_elbow[1]))
                            re = (int(right_elbow[0]), int(right_elbow[1]))
                            ns = (int(nose[0]), int(nose[1]))
                            
                            debug_info = {}
                            if (left_wrist_conf > 0.5 and right_wrist_conf > 0.5 and
                                left_shoulder_conf > 0.5 and right_shoulder_conf > 0.5 and
                                left_elbow_conf > 0.5 and right_elbow_conf > 0.5 and
                                nose_conf > 0.5):
                                
                                wrists_crossed = (lw[0] > rs[0] and rw[0] < ls[0] and lw[0] > rw[0])
                                debug_info["cruzados"] = wrists_crossed
                                arms_above_head = (lw[1] < ns[1] and rw[1] < ns[1])
                                debug_info["acima_cabeca"] = arms_above_head
                                wrist_distance_x = abs(lw[0] - rw[0])
                                close_horizontally = wrist_distance_x < frame_orig.shape[1] * 0.3
                                debug_info["proximos"] = close_horizontally
                                
                                pessoa_braco_cruzado = (wrists_crossed and arms_above_head and close_horizontally)
                                if pessoa_braco_cruzado:
                                    algum_braco_cruzado = True
                            else:
                                pessoa_braco_cruzado = False
                                debug_info = {"confianca_baixa": True}
                            
                            confidences = {
                                'left_wrist': left_wrist_conf > 0.5,
                                'right_wrist': right_wrist_conf > 0.5,
                                'left_shoulder': left_shoulder_conf > 0.5,
                                'right_shoulder': right_shoulder_conf > 0.5,
                                'left_elbow': left_elbow_conf > 0.5,
                                'right_elbow': right_elbow_conf > 0.5,
                                'nose': nose_conf > 0.5
                            }
                            
                            detected_people.append({
                                'keypoints': keypoints,
                                'braco_cruzado': pessoa_braco_cruzado,
                                'coords': {
                                    'left_wrist': lw,
                                    'right_wrist': rw,
                                    'left_shoulder': ls,
                                    'right_shoulder': rs,
                                    'left_elbow': le,
                                    'right_elbow': re,
                                    'nose': ns
                                },
                                'confidences': confidences,
                                'debug_info': debug_info
                            })
                        except Exception as e:
                            print(f"Erro ao processar pessoa: {e}")
                            continue
                            
                if algum_braco_cruzado:
                    if bracos_cruzados_start_time is None:
                        bracos_cruzados_start_time = current_time
                    elif current_time - bracos_cruzados_start_time >= bracos_cruzados_threshold:
                        grava = True
                        if not ultimo_estado_gravacao:
                            contador_bracos_cruzados += 1
                            pulse_led(3.0)  # aciona LED
                            ultimo_estado_gravacao = True
                else:
                    bracos_cruzados_start_time = None
                    grava = False
                    ultimo_estado_gravacao = False
                
                braco_cruzado_duracao = 0
                if bracos_cruzados_start_time is not None:
                    braco_cruzado_duracao = current_time - bracos_cruzados_start_time
                
                detection_result.pose_data = detected_people
                detection_result.braco_cruzado = algum_braco_cruzado
                detection_result.num_pessoas = num_pessoas
                detection_result.duracao_bracos_cruzados = braco_cruzado_duracao
                
            time.sleep(0.01)
        except Exception as e:
            print(f"Erro na thread de detecção: {e}")
            time.sleep(0.5)

# Funções para inicializar câmeras
def inicializar_picamera():
    global picam2
    from picamera2 import Picamera2
    if picam2 is not None:
        try:
            picam2.stop()
            picam2.close()
        except:
            pass
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (picam_width, picam_height)}
    )
    picam2.configure(config)
    picam2.start()
    print("PiCamera2 inicializada com resolução", picam_width, "x", picam_height)
    return True

def inicializar_webcam():
    global webcam
    if webcam is not None:
        try:
            webcam.release()
        except:
            pass
    webcam = cv2.VideoCapture(webcam_id)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
    if not webcam.isOpened():
        print("Erro ao abrir webcam!")
        return False
    print("Webcam inicializada com resolução", webcam_width, "x", webcam_height)
    return True

def alternar_camera():
    global usar_picamera
    with camera_lock:
        usar_picamera = not usar_picamera
        if usar_picamera:
            if webcam is not None:
                webcam.release()
            inicializar_picamera()
        else:
            if picam2 is not None:
                try:
                    picam2.stop()
                    picam2.close()
                except:
                    pass
            inicializar_webcam()
        print(f"Usando {'PiCamera' if usar_picamera else 'Webcam'} para captura")

def capturar_frame():
    frame = None
    with camera_lock:
        if usar_picamera:
            try:
                frame = picam2.capture_array()
            except Exception as e:
                print(f"Erro ao capturar frame da PiCamera: {e}")
        else:
            ret, frame = webcam.read()
            if not ret:
                print("Erro ao capturar frame da webcam")
                frame = None
    return frame

# Função para inicializar a câmera escolhida
if usar_picamera:
    inicializar_picamera()
else:
    inicializar_webcam()

# Inicia as threads de detecção e gravação
detection_th = threading.Thread(target=detection_thread, daemon=True)
detection_th.start()
video_writer_thd = threading.Thread(target=video_writer_thread, daemon=True)
video_writer_thd.start()

#Mostra ao usuario que o programa iniciou

print(f"Usando {'PiCamera' if usar_picamera else 'Webcam'} para captura de frames...")
print(f"Buffer configurado para {buffer_tamanho_segundos} segundos")
print(f"Detecção: {1/detection_interval:.1f} vezes por segundo")
print("Pressione 'q' para sair, 'g' para gravar manualmente, '+'/'-' para ajustar o buffer")
print("'v' para alternar visualização, 'd' para alternar diagnóstico, 'c' para alternar câmera")

frames_ok = 0
ultimo_frame_time = time.time()
tamanho_buffer = int(fps_medio * buffer_tamanho_segundos)
buffer_frames = deque(maxlen=tamanho_buffer)
buffer_timestamps = deque(maxlen=tamanho_buffer)
print(f"Tamanho inicial do buffer: {tamanho_buffer} frames")

while executando:
    frame_count += 1
    frame_rgb = capturar_frame()
    
    if frame_rgb is None:
        print("Falha ao capturar frame. Tentando novamente...")
        time.sleep(0.1)
        continue
    
    frame = cv2.flip(frame_rgb, 1)
    timestamp_atual = time.time()
    with buffer_lock:
        buffer_frames.append(frame.copy())
        buffer_timestamps.append(timestamp_atual)
    
    tempo_atual = time.time()
    frames_ok += 1
    if tempo_atual - ultimo_frame_time >= 1.0:
        fps_medio = frames_ok / (tempo_atual - ultimo_frame_time)
        novo_tamanho_buffer = int(fps_medio * buffer_tamanho_segundos)
        if abs(novo_tamanho_buffer - len(buffer_frames)) > fps_medio * 0.1:
            with buffer_lock:
                novo_buffer = deque(buffer_frames, maxlen=novo_tamanho_buffer)
                buffer_frames = novo_buffer
                novo_timestamps = deque(buffer_timestamps, maxlen=novo_tamanho_buffer)
                buffer_timestamps = novo_timestamps
            print(f"Buffer redimensionado: {novo_tamanho_buffer} frames ({buffer_tamanho_segundos} seg a {fps_medio:.1f} FPS)")
        frames_ok = 0
        ultimo_frame_time = tempo_atual
    
    frame_atual = frame
    frame_para_detectar = frame
    
    pose_data = detection_result.pose_data
    num_pessoas = detection_result.num_pessoas
    braco_cruzado_duracao = detection_result.duracao_bracos_cruzados
    
    if modo_visualizacao_completa:
        for person_data in pose_data:
            coords = person_data['coords']
            pessoa_braco_cruzado = person_data['braco_cruzado']
            confidences = person_data['confidences']
            
            if confidences['left_wrist']:
                cv2.circle(frame, coords['left_wrist'], 8, (255, 0, 0), -1)
            if confidences['right_wrist']:
                cv2.circle(frame, coords['right_wrist'], 8, (0, 255, 0), -1)
            if confidences['left_shoulder']:
                cv2.circle(frame, coords['left_shoulder'], 8, (0, 0, 255), -1)
            if confidences['right_shoulder']:
                cv2.circle(frame, coords['right_shoulder'], 8, (255, 255, 0), -1)
            if confidences['left_elbow']:
                cv2.circle(frame, coords['left_elbow'], 8, (0, 255, 255), -1)
            if confidences['right_elbow']:
                cv2.circle(frame, coords['right_elbow'], 8, (255, 0, 255), -1)
            if confidences['nose']:
                cv2.circle(frame, coords['nose'], 8, (255, 255, 255), -1)
            
            status_texto = "Bracos cruzados" if pessoa_braco_cruzado else "Bracos nao cruzados"
            if confidences['nose']:
                texto_pos = (coords['nose'][0] - 100, max(20, coords['nose'][1] - 30))
                cor_status = (0, 255, 0) if pessoa_braco_cruzado else (0, 0, 255)
                cv2.putText(frame, status_texto, texto_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_status, 2)
            
            if confidences['left_shoulder'] and confidences['left_elbow']:
                cv2.line(frame, coords['left_shoulder'], coords['left_elbow'], (255, 255, 255), 2)
            if confidences['left_elbow'] and confidences['left_wrist']:
                cv2.line(frame, coords['left_elbow'], coords['left_wrist'], (255, 255, 255), 2)
            if confidences['right_shoulder'] and confidences['right_elbow']:
                cv2.line(frame, coords['right_shoulder'], coords['right_elbow'], (255, 255, 255), 2)
            if confidences['right_elbow'] and confidences['right_wrist']:
                cv2.line(frame, coords['right_elbow'], coords['right_wrist'], (255, 255, 255), 2)
            if confidences['left_shoulder'] and confidences['right_shoulder']:
                cv2.line(frame, coords['left_shoulder'], coords['right_shoulder'], (255, 255, 255), 2)
            
            if mostrar_diagnostico and 'debug_info' in person_data:
                debug_info = person_data['debug_info']
                y_pos = 350
                for criterio, resultado in debug_info.items():
                    status = "SIM" if resultado else "NÃO"
                    cor = (0, 255, 0) if resultado else (0, 0, 255)
                    cv2.putText(frame, f"{criterio}: {status}", (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
                    y_pos += 30
            
        cv2.putText(frame, f"Grava: {grava}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if grava else (0, 0, 255), 2)
        cv2.putText(frame, f"Pessoas detectadas: {num_pessoas}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if bracos_cruzados_start_time is not None:
            cv2.putText(frame, f"Bracos cruzados: {braco_cruzado_duracao:.1f}s", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Deteccoes auto: {contador_bracos_cruzados}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Gravacoes manuais: {contador_gravacoes_manuais}", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
        cv2.putText(frame, f"Buffer: {len(buffer_frames)}/{buffer_frames.maxlen} frames", (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        cv2.putText(frame, f"FPS estimado: {fps_medio:.1f}", (10, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        cv2.putText(frame, f"Taxa de detecção: {1/detection_interval:.1f} vezes/s", (10, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        cv2.putText(frame, f"Câmera: {'PiCamera' if usar_picamera else 'Webcam'}", (10, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    else:
        for i, person_data in enumerate(pose_data):
            if person_data['braco_cruzado']:
                cv2.putText(frame, f"X", (20 + i*30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"O", (20 + i*30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Pessoas: {num_pessoas} | Gravando: {grava}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if grava else (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {fps_medio:.1f} | Det: {1/detection_interval:.1f}/s", (10, frame.shape[0]-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
        cv2.putText(frame, f"Câmera: {'PiCamera' if usar_picamera else 'Webcam'}", (10, frame.shape[0]-80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
        with buffer_lock:
            buffer_age = 0
            if len(buffer_timestamps) > 0:
                buffer_age = tempo_atual - buffer_timestamps[0]
            cv2.putText(frame, f"Buffer: {buffer_age:.0f}s", (10, frame.shape[0]-110),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
    
    cv2.putText(frame, "g: Gravar | +/-: Buffer | v: Vis. | c: Câmera | d: Diag. | q: Sair",
                (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    if gravando_video:
        cv2.putText(frame, "SALVANDO VIDEO...", (frame.shape[1]//2 - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    #Tela de exibicao, comentar abaixo se nao for debugar
    cv2.imshow("Deteccao de Bracos Cruzados (YOLOv8)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord('+'), ord('=')]:
        buffer_tamanho_segundos += 5
        novo_tamanho = int(fps_medio * buffer_tamanho_segundos)
        with buffer_lock:
            buffer_frames = deque(buffer_frames, maxlen=novo_tamanho)
            buffer_timestamps = deque(buffer_timestamps, maxlen=novo_tamanho)
        print(f"Buffer aumentado para {buffer_tamanho_segundos} segundos ({novo_tamanho} frames)")
    elif key in [ord('-'), ord('_')]:
        if buffer_tamanho_segundos > 5:
            buffer_tamanho_segundos -= 5
            novo_tamanho = int(fps_medio * buffer_tamanho_segundos)
            with buffer_lock:
                buffer_frames = deque(buffer_frames, maxlen=novo_tamanho)
                buffer_timestamps = deque(buffer_timestamps, maxlen=novo_tamanho)
            print(f"Buffer diminuído para {buffer_tamanho_segundos} segundos ({novo_tamanho} frames)")
    elif key == ord('g'):
        threading.Thread(target=salvar_buffer_como_video, args=(frame.copy(), True), daemon=True).start()
        print("Gravação manual iniciada")
    elif key == ord('v'):
        modo_visualizacao_completa = not modo_visualizacao_completa
        print(f"Modo de visualização {'completa' if modo_visualizacao_completa else 'simplificada'}")
    elif key == ord('d'):
        mostrar_diagnostico = not mostrar_diagnostico
        print(f"Diagnóstico de detecção {'ativado' if mostrar_diagnostico else 'desativado'}")
    elif key == ord('c'):
        alternar_camera()
    
    time.sleep(0.001)

print("Finalizando programa...")
executando = False
detection_th.join(timeout=1.0)
video_writer_thd.join(timeout=1.0)

with camera_lock:
    if picam2 is not None:
        try:
            picam2.stop()
            picam2.close()
        except:
            pass
    if webcam is not None:
        webcam.release()

cv2.destroyAllWindows()
print("Programa finalizado.")
