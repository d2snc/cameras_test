# -*- coding: utf-8 -*-
"""
Exemplo de detecção de braços cruzados com YOLOv8 + Picamera2
Adaptado de um código original que utilizava RTSP.
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

# Importações específicas da Picamera2
from picamera2 import Picamera2

# -- CONFIGURAÇÕES DE GRAVAÇÃO/DETALHES --
buffer_tamanho_segundos = 30  # Quantos segundos manter no buffer
pasta_gravacao = "gravacoes"  # Pasta onde os vídeos serão salvos
prefixo_arquivo = "braco_cruzado_"
prefixo_manual = "manual_"

# Cria pasta de gravação se não existir
if not os.path.exists(pasta_gravacao):
    os.makedirs(pasta_gravacao)

# -- VARIÁVEIS GLOBAIS --
frame_count = 0
grava = False          # Indica se deve gravar automaticamente (braços cruzados)
gravando_video = False # Indica se já está gravando um vídeo (para não gravar em paralelo)

# Controle de detecção e frequência
detection_interval = 0.2  # 5 detecções por segundo
bracos_cruzados_start_time = None
bracos_cruzados_threshold = 1.0  # 1 segundo para considerar braços cruzados

# Contadores e estados
contador_bracos_cruzados = 0
contador_gravacoes_manuais = 0
ultimo_estado_gravacao = False

# Variáveis para buffer circular (frames recentes)
buffer_frames = deque(maxlen=0)       # Será inicializado após conhecer o FPS
buffer_timestamps = deque(maxlen=0)   # Timestamps correspondentes

# Modelo YOLOv8 Pose
print("Carregando modelo YOLOv8...")
model = YOLO("yolov8n-pose.pt")  # Use yolov8m-pose.pt, yolov8l-pose.pt, etc., se desejar mais precisão

# Índices de keypoints (modelo COCO-pose)
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
NOSE = 0

# Classe simples para armazenar resultados de detecção
class DetectionResult:
    def __init__(self):
        self.pose_data = []
        self.braco_cruzado = False
        self.num_pessoas = 0
        self.duracao_bracos_cruzados = 0

# Variáveis compartilhadas entre threads
detection_result = DetectionResult()
frame_para_detectar = None
frame_atual = None
executando = True
buffer_lock = threading.Lock()

# Função para salvar o buffer de frames como vídeo
def salvar_buffer_como_video(trigger_frame=None, manual=False):
    global buffer_frames, buffer_timestamps, gravando_video, contador_gravacoes_manuais
    
    # Evita iniciar nova gravação se já estiver gravando
    if gravando_video:
        return
    
    gravando_video = True
    
    # Contador de gravações manuais
    if manual:
        contador_gravacoes_manuais += 1
    
    # Monta o nome do arquivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = prefixo_manual if manual else prefixo_arquivo
    nome_arquivo = os.path.join(pasta_gravacao, f"{prefix}{timestamp}.mp4")
    
    # Copia os frames do buffer (evita concorrência durante gravação)
    with buffer_lock:
        if len(buffer_frames) == 0:
            gravando_video = False
            return
        frames_para_gravar = list(buffer_frames)
        if trigger_frame is not None:
            frames_para_gravar.append(trigger_frame)
    
    if len(frames_para_gravar) == 0:
        gravando_video = False
        return
    
    # Dimensões do frame
    altura, largura = frames_para_gravar[0].shape[:2]
    
    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    video_writer = cv2.VideoWriter(nome_arquivo, fourcc, 30.0, (largura, altura))
    
    for frame in frames_para_gravar:
        video_writer.write(frame)
    video_writer.release()
    
    tipo_gravacao = "manual" if manual else "automática"
    print(f"Vídeo {tipo_gravacao} salvo: {nome_arquivo} ({len(frames_para_gravar)} frames)")
    
    gravando_video = False
    return nome_arquivo

# Thread que coordena quando salvar vídeos automaticamente
def video_writer_thread():
    global executando, grava, ultimo_estado_gravacao
    
    ultimo_estado_grava = False
    ultimo_arquivo_gravado_time = 0
    min_intervalo_gravacao = 5  # Segundos mínimos entre gravações automáticas
    
    while executando:
        if grava and not ultimo_estado_grava:
            tempo_atual = time.time()
            if tempo_atual - ultimo_arquivo_gravado_time >= min_intervalo_gravacao:
                salvar_buffer_como_video(frame_atual, manual=False)
                ultimo_arquivo_gravado_time = tempo_atual
        ultimo_estado_grava = grava
        time.sleep(0.1)

# Thread para rodar a detecção
def detection_thread():
    global detection_result, frame_para_detectar, executando
    global bracos_cruzados_start_time, contador_bracos_cruzados
    global ultimo_estado_gravacao, grava
    
    last_detection_time = 0
    
    while executando:
        current_time = time.time()
        
        # Faz detecção 10x/seg
        if current_time - last_detection_time >= detection_interval and frame_para_detectar is not None:
            frame_to_process = frame_para_detectar.copy()
            
            # Executa YOLOv8 Pose
            results = model(frame_to_process, verbose=False)
            last_detection_time = current_time
            
            algum_braco_cruzado = False
            num_pessoas = 0
            detected_people = []
            
            # Verifica se houve detecção
            if len(results) > 0 and len(results[0].keypoints.data) > 0:
                for person in results[0].keypoints.data:
                    num_pessoas += 1
                    keypoints = person.cpu().numpy()

                    # Extrai coordenadas
                    left_wrist = keypoints[LEFT_WRIST][:2]
                    right_wrist = keypoints[RIGHT_WRIST][:2]
                    left_shoulder = keypoints[LEFT_SHOULDER][:2]
                    right_shoulder = keypoints[RIGHT_SHOULDER][:2]
                    nose = keypoints[NOSE][:2]
                    
                    # Confianças
                    left_wrist_conf = keypoints[LEFT_WRIST][2]
                    right_wrist_conf = keypoints[RIGHT_WRIST][2]
                    left_shoulder_conf = keypoints[LEFT_SHOULDER][2]
                    right_shoulder_conf = keypoints[RIGHT_SHOULDER][2]
                    
                    # Converte para inteiro (x, y)
                    lw = (int(left_wrist[0]), int(left_wrist[1]))
                    rw = (int(right_wrist[0]), int(right_wrist[1]))
                    ls = (int(left_shoulder[0]), int(left_shoulder[1]))
                    rs = (int(right_shoulder[0]), int(right_shoulder[1]))
                    ns = (int(nose[0]), int(nose[1]))

                    # Valida pontos (confiança > 0.5)
                    if (left_wrist_conf > 0.5 and right_wrist_conf > 0.5 and 
                        left_shoulder_conf > 0.5 and right_shoulder_conf > 0.5):
                        
                        # Critério simples de "braços cruzados"
                        pessoa_braco_cruzado = (
                            lw[0] > rs[0] and
                            rw[0] < ls[0] and
                            lw[1] < ns[1] + 10 and
                            rw[1] < ns[1] + 10
                        )
                        
                        if pessoa_braco_cruzado:
                            algum_braco_cruzado = True
                        
                        detected_people.append({
                            'keypoints': keypoints,
                            'braco_cruzado': pessoa_braco_cruzado,
                            'coords': {
                                'left_wrist': lw,
                                'right_wrist': rw,
                                'left_shoulder': ls,
                                'right_shoulder': rs,
                                'nose': ns
                            }
                        })
            
            # Lógica de braço cruzado e gravação
            if algum_braco_cruzado:
                if bracos_cruzados_start_time is None:
                    bracos_cruzados_start_time = current_time
                elif current_time - bracos_cruzados_start_time >= bracos_cruzados_threshold:
                    grava = True
                    if not ultimo_estado_gravacao:
                        contador_bracos_cruzados += 1
                        ultimo_estado_gravacao = True
            else:
                bracos_cruzados_start_time = None
                grava = False
                ultimo_estado_gravacao = False
            
            # Duração braços cruzados
            braco_cruzado_duracao = 0
            if bracos_cruzados_start_time is not None:
                braco_cruzado_duracao = current_time - bracos_cruzados_start_time
            
            # Atualiza resultados globais
            detection_result.pose_data = detected_people
            detection_result.braco_cruzado = algum_braco_cruzado
            detection_result.num_pessoas = num_pessoas
            detection_result.duracao_bracos_cruzados = braco_cruzado_duracao
        
        time.sleep(0.005)

# ========== CONFIGURAÇÃO DA PICAMERA2 ===========
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (1280, 720)}
)
picam2.configure(config)
picam2.start()

# -- Inicia as threads --
detection_th = threading.Thread(target=detection_thread, daemon=True)
detection_th.start()

video_writer_thd = threading.Thread(target=video_writer_thread, daemon=True)
video_writer_thd.start()

print("Usando Picamera2 para captura de frames...")
print(f"Buffer configurado para {buffer_tamanho_segundos} segundos")
print("Deteccao: 5 vezes por segundo")
print("Pressione 'q' para sair, 'g' para gravar manualmente, '+'/'-' para ajustar o buffer")

# Variáveis para estimação de FPS
frames_ok = 0
ultimo_frame_time = time.time()
fps_medio = 30  # chute inicial de FPS
tamanho_buffer = int(fps_medio * buffer_tamanho_segundos)
buffer_frames = deque(maxlen=tamanho_buffer)
buffer_timestamps = deque(maxlen=tamanho_buffer)
print(f"Tamanho inicial do buffer: {tamanho_buffer} frames")

# Loop principal
while executando:
    frame_count += 1
    
    # Captura o frame em RGB
    frame_rgb = picam2.capture_array()
    if frame_rgb is None:
        continue
    
    # Espelha horizontalmente (opcional)
    frame = cv2.flip(frame_rgb, 1)
    
    # Adiciona no buffer circular (para gravação retroativa)
    timestamp_atual = time.time()
    with buffer_lock:
        buffer_frames.append(frame.copy())
        buffer_timestamps.append(timestamp_atual)
    
    # Calcula FPS real a cada segundo
    tempo_atual = time.time()
    frames_ok += 1
    if tempo_atual - ultimo_frame_time >= 1.0:
        fps_medio = frames_ok / (tempo_atual - ultimo_frame_time)
        # Ajusta buffer de acordo com FPS real
        novo_tamanho_buffer = int(fps_medio * buffer_tamanho_segundos)
        if abs(novo_tamanho_buffer - len(buffer_frames)) > fps_medio * 0.1:
            with buffer_lock:
                novo_buffer = deque(buffer_frames, maxlen=novo_tamanho_buffer)
                buffer_frames = novo_buffer
                novo_timestamps = deque(buffer_timestamps, maxlen=novo_tamanho_buffer)
                buffer_timestamps = novo_timestamps
            print(f"Buffer redimensionado: {novo_tamanho_buffer} frames "
                  f"({buffer_tamanho_segundos} seg a {fps_medio:.1f} FPS)")
        
        frames_ok = 0
        ultimo_frame_time = tempo_atual
    
    # Disponibiliza para thread de detecção
    frame_atual = frame.copy()
    frame_para_detectar = frame.copy()
    
    # Lê resultados de detecção
    pose_data = detection_result.pose_data
    num_pessoas = detection_result.num_pessoas
    braco_cruzado_duracao = detection_result.duracao_bracos_cruzados
    
    # Desenha os resultados
    for person_data in pose_data:
        coords = person_data['coords']
        pessoa_braco_cruzado = person_data['braco_cruzado']
        
        # Pontos de interesse
        cv2.circle(frame, coords['left_wrist'], 8, (255, 0, 0), -1)
        cv2.circle(frame, coords['right_wrist'], 8, (0, 255, 0), -1)
        cv2.circle(frame, coords['left_shoulder'], 8, (0, 0, 255), -1)
        cv2.circle(frame, coords['right_shoulder'], 8, (255, 255, 0), -1)
        cv2.circle(frame, coords['nose'], 8, (255, 0, 255), -1)
        
        status_texto = "Bracos cruzados" if pessoa_braco_cruzado else "Bracos nao cruzados"
        texto_pos = (coords['nose'][0] - 100, max(20, coords['nose'][1] - 30))
        cv2.putText(frame, status_texto, texto_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Linhas
        cv2.line(frame, coords['left_shoulder'], coords['left_wrist'], (255, 255, 255), 2)
        cv2.line(frame, coords['right_shoulder'], coords['right_wrist'], (255, 255, 255), 2)
        cv2.line(frame, coords['left_shoulder'], coords['right_shoulder'], (255, 255, 255), 2)
    
    # Mostra status
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
    cv2.putText(frame, f"Taxa de deteccao: 5 vezes/s", (10, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    
    cv2.putText(frame, "g: Gravar manual | +/-: Ajustar buffer | q: Sair", 
                (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    if gravando_video:
        cv2.putText(frame, "SALVANDO VIDEO...", (frame.shape[1]//2 - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    cv2.imshow("Deteccao de Bracos Cruzados (YOLOv8) - Picamera2", frame)
    
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
        # Gravação manual em thread separada
        threading.Thread(target=salvar_buffer_como_video, args=(frame.copy(), True)).start()
        print("Gravação manual iniciada")
    
    time.sleep(0.001)

# Finalização
executando = False
detection_th.join(timeout=1.0)
video_writer_thd.join(timeout=1.0)

picam2.stop()
picam2.close()
cv2.destroyAllWindows()
