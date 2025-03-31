# -*- coding: utf-8 -*-
"""
Exemplo de detecção de braços cruzados com YOLOv8 + Picamera2/Webcam
Adaptado e otimizado para Raspberry Pi com detecção aprimorada
Versão com correções de buffer e codec
"""
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import threading
import os
import subprocess
import serial
import time
from collections import deque
from datetime import datetime

#Abrindo conexão serial
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)

# Importações específicas da Picamera2 (mantidas, serão usadas condicionalmente)
from picamera2 import Picamera2

# -- CONFIGURAÇÕES DE GRAVAÇÃO/DETALHES --
buffer_tamanho_segundos = 30 # Quantos segundos manter no buffer
pasta_gravacao = "gravacoes" # Pasta onde os vídeos serão salvos
prefixo_arquivo = "braco_cruzado_"
prefixo_manual = "manual_"

# -- CONFIGURAÇÃO DE CÂMERA --
usar_picamera = True  # Flag para escolher entre Picamera ou Webcam
webcam_id = 0  # ID da webcam (normalmente 0 para a primeira webcam)
webcam_width = 1280  # Largura para webcam
webcam_height = 720  # Altura para webcam
picam_width = 1280  # Largura para PiCamera 
picam_height = 720  # Altura para PiCamera

# Cria pasta de gravação se não existir
if not os.path.exists(pasta_gravacao):
    os.makedirs(pasta_gravacao)

# -- VARIÁVEIS GLOBAIS --
frame_count = 0
grava = False # Indica se deve gravar automaticamente (braços cruzados)
gravando_video = False # Indica se já está gravando um vídeo (para não gravar em paralelo)

# Controle de detecção e frequência
detection_interval = 0.2  # [OTIMIZAÇÃO 3] 2 detecções por segundo em vez de 5
bracos_cruzados_start_time = None
bracos_cruzados_threshold = 1.0 # 1 segundo para considerar braços cruzados

# Contadores e estados
contador_bracos_cruzados = 0
contador_gravacoes_manuais = 0
ultimo_estado_gravacao = False

# Variáveis para buffer circular (frames recentes)
buffer_frames = deque(maxlen=0) # Será inicializado após conhecer o FPS
buffer_timestamps = deque(maxlen=0) # Timestamps correspondentes

# [OTIMIZAÇÃO 7] Modo de visualização reduzida para menor processamento
modo_visualizacao_completa = True  # Pode ser alterado para False para menos processamento gráfico
mostrar_diagnostico = True  # Mostrar informações de diagnóstico da detecção

# [OTIMIZAÇÃO 2] Modelo YOLOv8 - mantém o mais leve mas deixa comentado alternativas de otimização
print("Carregando modelo YOLOv8...")
model = YOLO("yolov8n-pose.pt")  # Modelo mais leve
# Alternativas:
# model = YOLO("yolov8n-pose_openvino_model")  # Se convertido para OpenVINO
# model = YOLO("yolov8n-pose.engine")  # Se convertido para TensorRT

# Índices de keypoints (modelo COCO-pose)
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7  # Adicionado para melhor detecção de braços cruzados
RIGHT_ELBOW = 8  # Adicionado para melhor detecção de braços cruzados
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
camera_lock = threading.Lock()
fps_medio = 30  # Valor inicial, será atualizado durante a execução

# Variáveis de captura
picam2 = None
webcam = None

# [SOLUÇÃO 4] Thread dedicada com prioridade para gravação de vídeo
def thread_gravacao_video_prioridade(frames_para_gravar, nome_arquivo, fps_gravacao, altura, largura, manual=False):
    global gravando_video, contador_gravacoes_manuais
    
    try:
        # Tenta aumentar a prioridade (não se preocupe se falhar)
        os.nice(-10)
    except:
        pass
    
    #Chama o flash para alertar o usuário
    message = "F" #Pode ser qq msg
    ser.write(message.encode()) #Envia a msg
    
    # Usar codec XVID que é mais compatível com Raspberry Pi
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(nome_arquivo, fourcc, fps_gravacao, (largura, altura))
    print(f"Usando codec XVID para gravar {len(frames_para_gravar)} frames")
    
    # Verifica se o VideoWriter foi inicializado corretamente
    if not video_writer.isOpened():
        print(f"Erro: Não foi possível inicializar o VideoWriter para {nome_arquivo}")
        gravando_video = False
        return False
    
    # Grava cada frame no vídeo com contagem para debug
    frame_count = 0
    start_time = time.time()
    
    print(f"Iniciando gravação de {len(frames_para_gravar)} frames...")
    
    # Garanta que estamos gravando todos os frames do buffer
    for i, frame in enumerate(frames_para_gravar):
        if frame is not None:
            video_writer.write(frame)
            frame_count += 1
            # Reporta progresso de 10 em 10%
            if len(frames_para_gravar) > 10 and i % (len(frames_para_gravar) // 10) == 0:
                print(f"Progresso: {i / len(frames_para_gravar) * 100:.1f}% ({i}/{len(frames_para_gravar)} frames)")
    
    # Finaliza a gravação
    video_writer.release()
    
    # Calcula o tempo de gravação
    tempo_gravacao = time.time() - start_time
    tipo_gravacao = "manual" if manual else "automática"
    print(f"Vídeo {tipo_gravacao} salvo: {nome_arquivo}")
    print(f"Gravados {frame_count} frames de {len(frames_para_gravar)} em {tempo_gravacao:.2f} segundos")
    print(f"Taxa de gravação: {frame_count/tempo_gravacao:.2f} fps")
    
    gravando_video = False
    return True

# [SOLUÇÃO 1, 2 e 4] Função para salvar o buffer de frames com FPS correto e thread dedicada
def salvar_buffer_como_video(trigger_frame=None, manual=False):
    global buffer_frames, buffer_timestamps, gravando_video, contador_gravacoes_manuais, fps_medio
    
    # Evita iniciar nova gravação se já estiver gravando
    if gravando_video:
        print("Já existe uma gravação em andamento. Aguarde...")
        return
    
    gravando_video = True
    
    # Contador de gravações manuais
    if manual:
        contador_gravacoes_manuais += 1
    
    # Monta o nome do arquivo (com extensão AVI para maior compatibilidade)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = prefixo_manual if manual else prefixo_arquivo
    nome_arquivo = os.path.join(pasta_gravacao, f"{prefix}{timestamp}.avi")
    
    # DEBUG: Verificar tamanho do buffer antes de copiar
    print(f"Tamanho do buffer antes da cópia: {len(buffer_frames)} frames")
    
    # Copia os frames do buffer (evita concorrência durante gravação)
    frames_para_gravar = []
    with buffer_lock:
        if len(buffer_frames) == 0:
            print("ERRO: Buffer vazio, nada para gravar")
            gravando_video = False
            return
        
        # Fazer uma cópia explícita de cada frame para evitar referências compartilhadas
        for frame in buffer_frames:
            frames_para_gravar.append(frame.copy())
        
        # Adicionar o frame atual se fornecido
        if trigger_frame is not None:
            frames_para_gravar.append(trigger_frame.copy())
    
    # DEBUG: Verificar quantidade de frames após cópia
    print(f"Frames para gravar após cópia do buffer: {len(frames_para_gravar)}")
    
    if len(frames_para_gravar) == 0:
        print("ERRO: Nenhum frame copiado para gravação")
        gravando_video = False
        return
    
    # Dimensões do frame
    altura, largura = frames_para_gravar[0].shape[:2]
    
    # [SOLUÇÃO 1] Usa o FPS real medido para a gravação
    fps_gravacao = max(10.0, min(fps_medio, 30.0))  # Limita entre 10 e 30 fps
    
    print(f"Iniciando gravação com {len(frames_para_gravar)} frames a {fps_gravacao:.1f} fps...")
    print(f"Buffer configurado para {buffer_tamanho_segundos} segundos")
    
    # Criar uma thread dedicada para a gravação
    threading.Thread(
        target=thread_gravacao_video_prioridade,
        args=(frames_para_gravar, nome_arquivo, fps_gravacao, altura, largura, manual),
        daemon=True
    ).start()
    
    return nome_arquivo

# Thread que coordena quando salvar vídeos automaticamente
def video_writer_thread():
    global executando, grava, ultimo_estado_gravacao
    ultimo_estado_grava = False
    ultimo_arquivo_gravado_time = 0
    min_intervalo_gravacao = 5 # Segundos mínimos entre gravações automáticas
    while executando:
        if grava and not ultimo_estado_grava:
            tempo_atual = time.time()
            if tempo_atual - ultimo_arquivo_gravado_time >= min_intervalo_gravacao:
                # DIAGNÓSTICO: Verificar o tamanho do buffer antes de iniciar gravação
                with buffer_lock:
                    print(f"Estado do buffer no momento da detecção: {len(buffer_frames)} frames")
                    if len(buffer_frames) > 0:
                        buffer_age = tempo_atual - buffer_timestamps[0]
                        print(f"Buffer contém frames de {buffer_age:.1f} segundos atrás")
                
                salvar_buffer_como_video(frame_atual, manual=False)
                ultimo_arquivo_gravado_time = tempo_atual
        
        ultimo_estado_grava = grava
        time.sleep(0.1)

# [OTIMIZAÇÃO 4] Thread para rodar a detecção com frames reduzidos
def detection_thread():
    global detection_result, frame_para_detectar, executando
    global bracos_cruzados_start_time, contador_bracos_cruzados
    global ultimo_estado_gravacao, grava
    last_detection_time = 0
    
    # Dimensão reduzida para processamento
    detection_width, detection_height = 640, 480
    
    while executando:
        current_time = time.time()
        # [OTIMIZAÇÃO 3] Faz detecção com frequência reduzida
        if current_time - last_detection_time >= detection_interval and frame_para_detectar is not None:
            # [OTIMIZAÇÃO 4] Redimensionar para processamento mais rápido
            frame_orig = frame_para_detectar.copy()
            frame_to_process = cv2.resize(frame_orig, (detection_width, detection_height))
            
            # Executa YOLOv8 Pose no frame reduzido
            results = model(frame_to_process, verbose=False)
            last_detection_time = current_time
            
            algum_braco_cruzado = False
            num_pessoas = 0
            detected_people = []
            
            # Verifica se houve detecção
            if len(results) > 0 and len(results[0].keypoints.data) > 0:
                # Fator de escala para mapear coordenadas de volta ao frame original
                scale_x = frame_orig.shape[1] / detection_width
                scale_y = frame_orig.shape[0] / detection_height
                
                for person in results[0].keypoints.data:
                    num_pessoas += 1
                    keypoints = person.cpu().numpy()
                    
                    # Extrai coordenadas e escala de volta ao tamanho original
                    left_wrist = keypoints[LEFT_WRIST][:2] * np.array([scale_x, scale_y])
                    right_wrist = keypoints[RIGHT_WRIST][:2] * np.array([scale_x, scale_y])
                    left_shoulder = keypoints[LEFT_SHOULDER][:2] * np.array([scale_x, scale_y])
                    right_shoulder = keypoints[RIGHT_SHOULDER][:2] * np.array([scale_x, scale_y])
                    left_elbow = keypoints[LEFT_ELBOW][:2] * np.array([scale_x, scale_y])
                    right_elbow = keypoints[RIGHT_ELBOW][:2] * np.array([scale_x, scale_y])
                    nose = keypoints[NOSE][:2] * np.array([scale_x, scale_y])
                    
                    # Confianças
                    left_wrist_conf = keypoints[LEFT_WRIST][2]
                    right_wrist_conf = keypoints[RIGHT_WRIST][2]
                    left_shoulder_conf = keypoints[LEFT_SHOULDER][2]
                    right_shoulder_conf = keypoints[RIGHT_SHOULDER][2]
                    left_elbow_conf = keypoints[LEFT_ELBOW][2]
                    right_elbow_conf = keypoints[RIGHT_ELBOW][2]
                    nose_conf = keypoints[NOSE][2]
                    
                    # Converte para inteiro (x, y)
                    lw = (int(left_wrist[0]), int(left_wrist[1]))
                    rw = (int(right_wrist[0]), int(right_wrist[1]))
                    ls = (int(left_shoulder[0]), int(left_shoulder[1]))
                    rs = (int(right_shoulder[0]), int(right_shoulder[1]))
                    le = (int(left_elbow[0]), int(left_elbow[1]))
                    re = (int(right_elbow[0]), int(right_elbow[1]))
                    ns = (int(nose[0]), int(nose[1]))
                    
                    # Lógica de detecção de braços cruzados corrigida para ACIMA da cabeça
                    debug_info = {}  # Dicionário para armazenar informações de diagnóstico
                    
                    if (left_wrist_conf > 0.5 and right_wrist_conf > 0.5 and
                        left_shoulder_conf > 0.5 and right_shoulder_conf > 0.5 and
                        left_elbow_conf > 0.5 and right_elbow_conf > 0.5 and
                        nose_conf > 0.5):
                        
                        # Verifica se os pulsos estão cruzados horizontalmente
                        wrists_crossed = (
                            lw[0] > rs[0] and       # Pulso esquerdo à direita do ombro direito
                            rw[0] < ls[0] and       # Pulso direito à esquerda do ombro esquerdo
                            lw[0] > rw[0]           # Pulso esquerdo à direita do pulso direito
                        )
                        debug_info["cruzados"] = wrists_crossed
                        
                        # CORREÇÃO: Verifica se os braços estão acima da cabeça (em vez de na altura do peito)
                        arms_above_head = (
                            lw[1] < ns[1] and       # Pulsos acima do nariz
                            rw[1] < ns[1]
                        )
                        debug_info["acima_cabeca"] = arms_above_head
                        
                        # Verifica distância horizontal entre os pulsos (para garantir que estão próximos o suficiente)
                        wrist_distance_x = abs(lw[0] - rw[0])
                        close_horizontally = wrist_distance_x < frame_orig.shape[1] * 0.3  # 30% da largura da tela
                        debug_info["proximos"] = close_horizontally
                        
                        # Combinação de todas as condições
                        pessoa_braco_cruzado = (
                            wrists_crossed and 
                            arms_above_head and
                            close_horizontally
                        )
                        
                        if pessoa_braco_cruzado:
                            algum_braco_cruzado = True
                    else:
                        pessoa_braco_cruzado = False
                        debug_info = {
                            "confianca_baixa": True
                        }
                    
                    # Armazena informações de confiança também para uso no desenho
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
        time.sleep(0.01)  # Reduz uso da CPU

# Função para inicializar a PiCamera
def inicializar_picamera():
    global picam2
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

# Função para inicializar a Webcam
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

# Função para alternar entre câmeras
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

# Função para capturar um frame da câmera atual
def capturar_frame():
    frame = None
    with camera_lock:
        if usar_picamera and picam2 is not None:
            try:
                frame = picam2.capture_array()
            except Exception as e:
                print(f"Erro ao capturar frame da PiCamera: {e}")
        elif not usar_picamera and webcam is not None:
            ret, frame = webcam.read()
            if not ret:
                print("Erro ao capturar frame da webcam")
                frame = None
    
    return frame

# Inicializar a câmera escolhida
if usar_picamera:
    inicializar_picamera()
else:
    inicializar_webcam()

# -- Inicia as threads --
detection_th = threading.Thread(target=detection_thread, daemon=True)
detection_th.start()
video_writer_thd = threading.Thread(target=video_writer_thread, daemon=True)
video_writer_thd.start()

print(f"Usando {'PiCamera' if usar_picamera else 'Webcam'} para captura de frames...")
print(f"Buffer configurado para {buffer_tamanho_segundos} segundos")
print(f"Detecção: {1/detection_interval:.1f} vezes por segundo")
print("Pressione 'q' para sair, 'g' para gravar manualmente, '+'/'-' para ajustar o buffer")
print("'v' para alternar visualização, 'd' para alternar diagnóstico, 'c' para alternar câmera")

# Variáveis para estimação de FPS
frames_ok = 0
ultimo_frame_time = time.time()
tamanho_buffer = int(fps_medio * buffer_tamanho_segundos)
buffer_frames = deque(maxlen=tamanho_buffer)
buffer_timestamps = deque(maxlen=tamanho_buffer)
print(f"Tamanho inicial do buffer: {tamanho_buffer} frames")

# Loop principal
while executando:
    frame_count += 1
    
    # Captura o frame da câmera ativa
    frame_rgb = capturar_frame()
    
    if frame_rgb is None:
        print("Falha ao capturar frame. Tentando novamente...")
        time.sleep(0.1)
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
        
        # Verificação periódica da saúde do buffer
        if frames_ok == 0 and frame_count % 300 == 0:  # A cada 300 frames, faça uma verificação
            with buffer_lock:
                buffer_age = 0
                if len(buffer_timestamps) > 0:
                    buffer_age = time.time() - buffer_timestamps[0]
                print(f"Verificação de buffer: {len(buffer_frames)} frames, idade mais antiga: {buffer_age:.1f} segundos")
        
        frames_ok = 0
        ultimo_frame_time = tempo_atual
    
    # Disponibiliza para thread de detecção
    frame_atual = frame
    frame_para_detectar = frame
    
    # Lê resultados de detecção
    pose_data = detection_result.pose_data
    num_pessoas = detection_result.num_pessoas
    braco_cruzado_duracao = detection_result.duracao_bracos_cruzados
    
    # [OTIMIZAÇÃO 7] Desenha os resultados conforme modo de visualização
    if modo_visualizacao_completa:
        # Desenho completo - modo original
        for person_data in pose_data:
            coords = person_data['coords']
            pessoa_braco_cruzado = person_data['braco_cruzado']
            confidences = person_data['confidences']  # Usamos para verificar antes de desenhar
            
            # Pontos de interesse - desenha apenas se confiança for boa
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
                cv2.putText(frame, status_texto, texto_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_status, 2)
            
            # CORREÇÃO: Desenhar linhas apenas quando ambos os pontos têm confiança alta
            # Linha do ombro esquerdo ao cotovelo esquerdo
            if confidences['left_shoulder'] and confidences['left_elbow']:
                cv2.line(frame, coords['left_shoulder'], coords['left_elbow'], (255, 255, 255), 2)
            
            # Linha do cotovelo esquerdo ao pulso esquerdo
            if confidences['left_elbow'] and confidences['left_wrist']:
                cv2.line(frame, coords['left_elbow'], coords['left_wrist'], (255, 255, 255), 2)
            
            # Linha do ombro direito ao cotovelo direito
            if confidences['right_shoulder'] and confidences['right_elbow']:
                cv2.line(frame, coords['right_shoulder'], coords['right_elbow'], (255, 255, 255), 2)
            
            # Linha do cotovelo direito ao pulso direito
            if confidences['right_elbow'] and confidences['right_wrist']:
                cv2.line(frame, coords['right_elbow'], coords['right_wrist'], (255, 255, 255), 2)
            
            # Linha entre os ombros
            if confidences['left_shoulder'] and confidences['right_shoulder']:
                cv2.line(frame, coords['left_shoulder'], coords['right_shoulder'], (255, 255, 255), 2)
            
            # Mostrar diagnóstico de detecção (para a primeira pessoa apenas)
            if mostrar_diagnostico and 'debug_info' in person_data:
                debug_info = person_data['debug_info']
                y_pos = 350
                for criterio, resultado in debug_info.items():
                    status = "SIM" if resultado else "NÃO"
                    cor = (0, 255, 0) if resultado else (0, 0, 255)
                    cv2.putText(frame, f"{criterio}: {status}", (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
                    y_pos += 30
            
        # Mostra status detalhado
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
        # Exibir a câmera atualmente em uso
        cv2.putText(frame, f"Câmera: {'PiCamera' if usar_picamera else 'Webcam'}", (10, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    else:
        # [OTIMIZAÇÃO 7] Modo de visualização simplificada - menos processamento
        # Indicador simples de pessoas
        for i, person_data in enumerate(pose_data):
            if person_data['braco_cruzado']:
                cv2.putText(frame, f"X", (20 + i*30, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"O", (20 + i*30, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Informações principais
        cv2.putText(frame, f"Pessoas: {num_pessoas} | Gravando: {grava}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if grava else (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {fps_medio:.1f} | Det: {1/detection_interval:.1f}/s", (10, frame.shape[0]-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
        # Exibir a câmera atual
        cv2.putText(frame, f"Câmera: {'PiCamera' if usar_picamera else 'Webcam'}", (10, frame.shape[0]-80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
        
        # Buffer status simplificado
        with buffer_lock:
            buffer_age = 0
            if len(buffer_timestamps) > 0:
                buffer_age = tempo_atual - buffer_timestamps[0]
            cv2.putText(frame, f"Buffer: {buffer_age:.0f}s", (10, frame.shape[0]-110),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
    
    # Mensagens comuns a ambos os modos
    cv2.putText(frame, "g: Gravar | +/-: Buffer | v: Vis. | c: Câmera | d: Diag. | q: Sair",
                (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    if gravando_video:
        cv2.putText(frame, "SALVANDO VIDEO...", (frame.shape[1]//2 - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    #cv2.imshow("Deteccao de Bracos Cruzados (YOLOv8)", frame)
    
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
    elif key == ord('v'):
        # [OTIMIZAÇÃO 7] Alterna modo de visualização
        modo_visualizacao_completa = not modo_visualizacao_completa
        print(f"Modo de visualização {'completa' if modo_visualizacao_completa else 'simplificada'}")
    elif key == ord('d'):
        # Alterna exibição de diagnóstico
        mostrar_diagnostico = not mostrar_diagnostico
        print(f"Diagnóstico de detecção {'ativado' if mostrar_diagnostico else 'desativado'}")
    elif key == ord('c'):
        # Alternar entre PiCamera e Webcam
        alternar_camera()
    
    time.sleep(0.001)

# Finalização
print("Finalizando programa...")
executando = False
detection_th.join(timeout=1.0)
video_writer_thd.join(timeout=1.0)

# Liberar recursos de câmera
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
