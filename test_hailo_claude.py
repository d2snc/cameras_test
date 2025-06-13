#!/usr/bin/env python3
"""
Sistema de Detecção de Braços Cruzados com Hailo AI
Baseado no GStreamer e Hailo para melhor desempenho no Raspberry Pi 5
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import time
import threading
from collections import deque
from datetime import datetime

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# -- CONFIGURAÇÕES DE GRAVAÇÃO --
BUFFER_TAMANHO_SEGUNDOS = 50  # Segundos no buffer
PASTA_GRAVACAO = "gravacoes"
PREFIXO_ARQUIVO = "braco_cruzado_"
PREFIXO_MANUAL = "manual_"
BRACOS_CRUZADOS_THRESHOLD = 0.8  # Segundos para considerar braços cruzados
FPS_GRAVACAO = 30.0

# -- CONFIGURAÇÕES DE DETECÇÃO --
DETECTION_INTERVAL = 0.2  # 5 detecções por segundo

# Cria pasta de gravação
if not os.path.exists(PASTA_GRAVACAO):
    os.makedirs(PASTA_GRAVACAO)

# -- Índices de keypoints COCO --
KEYPOINTS = {
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

# ───────── GPIO via libgpiod ───────── #
try:
    import gpiod
    CHIP_NAME = "gpiochip0"
    LED_LINE_OFFSET = 17
    chip = gpiod.Chip(CHIP_NAME)
    led_line = chip.get_line(LED_LINE_OFFSET)
    led_line.request(
        consumer="hailo-led",
        type=gpiod.LINE_REQ_DIR_OUT,
        default_vals=[0],
    )
    LED_AVAILABLE = True
except Exception as e:
    print(f"LED desativado (libgpiod indisponível): {e}")
    LED_AVAILABLE = False

# -----------------------------------------------------------------------------------------------
# Classe customizada para gerenciar o estado da aplicação
# -----------------------------------------------------------------------------------------------
class HailoArmsDetectionApp(app_callback_class):
    def __init__(self):
        super().__init__()
        self.use_frame = True  # Sempre processar frames
        
        # Variáveis de estado
        self.frame_count = 0
        self.grava = False
        self.gravando_video = False
        self.bracos_cruzados_start_time = None
        self.contador_bracos_cruzados = 0
        self.contador_gravacoes_manuais = 0
        self.ultimo_estado_gravacao = False
        self.last_detection_time = 0
        
        # Buffer circular
        self.buffer_frames = deque(maxlen=int(FPS_GRAVACAO * BUFFER_TAMANHO_SEGUNDOS))
        self.buffer_timestamps = deque(maxlen=int(FPS_GRAVACAO * BUFFER_TAMANHO_SEGUNDOS))
        self.buffer_lock = threading.Lock()
        
        # Variáveis de visualização
        self.modo_visualizacao_completa = True
        self.mostrar_diagnostico = True
        self.current_frame = None
        self.detection_results = []
        
        # Thread de gravação
        self.executando = True
        self.video_writer_thread = threading.Thread(target=self._video_writer_thread, daemon=True)
        self.video_writer_thread.start()
        
        # Controle de teclado
        self.setup_keyboard_handler()
        
    def setup_keyboard_handler(self):
        """Configura thread para capturar teclas"""
        self.keyboard_thread = threading.Thread(target=self._keyboard_handler, daemon=True)
        self.keyboard_thread.start()
        
    def _keyboard_handler(self):
        """Thread para processar comandos do teclado"""
        import sys, tty, termios
        
        # Salva configurações do terminal
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            # Configura terminal para modo raw
            tty.setraw(sys.stdin.fileno())
            
            print("\nComandos disponíveis:")
            print("q: Sair | g: Gravar manual | v: Alternar visualização")
            print("d: Alternar diagnóstico | +/-: Ajustar buffer\n")
            
            while self.executando:
                key = sys.stdin.read(1)
                
                if key == 'q':
                    self.executando = False
                    break
                elif key == 'g':
                    self.gravar_manual()
                elif key == 'v':
                    self.modo_visualizacao_completa = not self.modo_visualizacao_completa
                    print(f"\nVisualização {'completa' if self.modo_visualizacao_completa else 'simplificada'}")
                elif key == 'd':
                    self.mostrar_diagnostico = not self.mostrar_diagnostico
                    print(f"\nDiagnóstico {'ativado' if self.mostrar_diagnostico else 'desativado'}")
                elif key in ['+', '=']:
                    self.ajustar_buffer(5)
                elif key in ['-', '_']:
                    self.ajustar_buffer(-5)
                    
        finally:
            # Restaura configurações do terminal
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    def pulse_led(self, duration=3.0):
        """Acende o LED por 'duration' segundos"""
        if not LED_AVAILABLE:
            return
            
        def _pulse():
            led_line.set_value(1)
            time.sleep(duration)
            led_line.set_value(0)
            
        threading.Thread(target=_pulse, daemon=True).start()
    
    def detectar_bracos_cruzados(self, keypoints, bbox, width, height):
        """Detecta se a pessoa está com os braços cruzados acima da cabeça"""
        try:
            # Extrai coordenadas dos keypoints necessários
            points = keypoints[0].get_points()
            
            # Converte coordenadas normalizadas para pixels
            def get_point_coords(point_name):
                idx = KEYPOINTS[point_name]
                if idx < len(points):
                    point = points[idx]
                    x = int((point.x() * bbox.width() + bbox.xmin()) * width)
                    y = int((point.y() * bbox.height() + bbox.ymin()) * height)
                    return (x, y), point.confidence() if hasattr(point, 'confidence') else 1.0
                return None, 0.0
            
            # Obtém coordenadas
            lw, lw_conf = get_point_coords('left_wrist')
            rw, rw_conf = get_point_coords('right_wrist')
            ls, ls_conf = get_point_coords('left_shoulder')
            rs, rs_conf = get_point_coords('right_shoulder')
            nose, nose_conf = get_point_coords('nose')
            
            # Verifica se todos os pontos necessários foram detectados com confiança
            min_confidence = 0.5
            if not all([lw, rw, ls, rs, nose]) or \
               any([lw_conf < min_confidence, rw_conf < min_confidence,
                    ls_conf < min_confidence, rs_conf < min_confidence,
                    nose_conf < min_confidence]):
                return False, {}
            
            # Critérios para braços cruzados
            wrists_crossed = (lw[0] > rs[0] and rw[0] < ls[0] and lw[0] > rw[0])
            arms_above_head = (lw[1] < nose[1] and rw[1] < nose[1])
            wrist_distance_x = abs(lw[0] - rw[0])
            close_horizontally = wrist_distance_x < width * 0.3
            
            # Debug info
            debug_info = {
                "cruzados": wrists_crossed,
                "acima_cabeca": arms_above_head,
                "proximos": close_horizontally
            }
            
            # Resultado final
            bracos_cruzados = wrists_crossed and arms_above_head and close_horizontally
            
            return bracos_cruzados, debug_info
            
        except Exception as e:
            print(f"Erro na detecção: {e}")
            return False, {}
    
    def salvar_buffer_como_video(self, trigger_frame=None, manual=False):
        """Salva o buffer de frames como vídeo"""
        if self.gravando_video:
            print("Já existe uma gravação em andamento...")
            return
            
        self.gravando_video = True
        
        if manual:
            self.contador_gravacoes_manuais += 1
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = PREFIXO_MANUAL if manual else PREFIXO_ARQUIVO
        nome_arquivo = os.path.join(PASTA_GRAVACAO, f"{prefix}{timestamp}.avi")
        
        # Copia frames do buffer
        frames_para_gravar = []
        with self.buffer_lock:
            if len(self.buffer_frames) == 0:
                print("Buffer vazio!")
                self.gravando_video = False
                return
                
            for frame in self.buffer_frames:
                frames_para_gravar.append(frame.copy())
            if trigger_frame is not None:
                frames_para_gravar.append(trigger_frame.copy())
        
        print(f"Salvando {len(frames_para_gravar)} frames...")
        
        # Thread de gravação
        def _gravar():
            try:
                altura, largura = frames_para_gravar[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(nome_arquivo, fourcc, FPS_GRAVACAO, (largura, altura))
                
                for frame in frames_para_gravar:
                    video_writer.write(frame)
                    
                video_writer.release()
                tipo = "manual" if manual else "automática"
                print(f"Vídeo {tipo} salvo: {nome_arquivo}")
                
            except Exception as e:
                print(f"Erro ao gravar vídeo: {e}")
            finally:
                self.gravando_video = False
                
        threading.Thread(target=_gravar, daemon=True).start()
    
    def _video_writer_thread(self):
        """Thread que coordena gravação automática"""
        ultimo_estado_grava = False
        ultimo_arquivo_gravado_time = 0
        min_intervalo_gravacao = 5  # Segundos entre gravações
        
        while self.executando:
            if self.grava and not ultimo_estado_grava:
                tempo_atual = time.time()
                if tempo_atual - ultimo_arquivo_gravado_time >= min_intervalo_gravacao:
                    self.salvar_buffer_como_video(self.current_frame, manual=False)
                    ultimo_arquivo_gravado_time = tempo_atual
                    
            ultimo_estado_grava = self.grava
            time.sleep(0.1)
    
    def gravar_manual(self):
        """Inicia gravação manual"""
        threading.Thread(
            target=self.salvar_buffer_como_video,
            args=(self.current_frame.copy() if self.current_frame is not None else None, True),
            daemon=True
        ).start()
        print("\nGravação manual iniciada")
    
    def ajustar_buffer(self, delta):
        """Ajusta tamanho do buffer"""
        global BUFFER_TAMANHO_SEGUNDOS
        novo_tamanho = BUFFER_TAMANHO_SEGUNDOS + delta
        if novo_tamanho >= 5:
            BUFFER_TAMANHO_SEGUNDOS = novo_tamanho
            novo_max = int(FPS_GRAVACAO * BUFFER_TAMANHO_SEGUNDOS)
            with self.buffer_lock:
                self.buffer_frames = deque(self.buffer_frames, maxlen=novo_max)
                self.buffer_timestamps = deque(self.buffer_timestamps, maxlen=novo_max)
            print(f"\nBuffer ajustado para {BUFFER_TAMANHO_SEGUNDOS} segundos")
    
    def draw_visualization(self, frame):
        """Desenha visualização no frame"""
        if not self.modo_visualizacao_completa:
            # Visualização simplificada
            cv2.putText(frame, f"Pessoas: {len(self.detection_results)} | Gravando: {self.grava}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if self.grava else (0, 0, 255), 2)
            return frame
            
        # Visualização completa
        for result in self.detection_results:
            # Desenha keypoints
            for name, (x, y) in result['keypoints'].items():
                color = (0, 255, 0) if result['bracos_cruzados'] else (255, 0, 0)
                cv2.circle(frame, (x, y), 5, color, -1)
                
            # Desenha conexões
            connections = [
                ('left_shoulder', 'left_elbow'),
                ('left_elbow', 'left_wrist'),
                ('right_shoulder', 'right_elbow'),
                ('right_elbow', 'right_wrist'),
                ('left_shoulder', 'right_shoulder')
            ]
            
            for start, end in connections:
                if start in result['keypoints'] and end in result['keypoints']:
                    cv2.line(frame, result['keypoints'][start], 
                            result['keypoints'][end], (255, 255, 255), 2)
            
            # Status texto
            if 'nose' in result['keypoints']:
                status = "Braços cruzados" if result['bracos_cruzados'] else "Normal"
                pos = (result['keypoints']['nose'][0] - 50, result['keypoints']['nose'][1] - 20)
                color = (0, 255, 0) if result['bracos_cruzados'] else (0, 0, 255)
                cv2.putText(frame, status, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Informações gerais
        cv2.putText(frame, f"Grava: {self.grava}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if self.grava else (0, 0, 255), 2)
        cv2.putText(frame, f"Pessoas: {len(self.detection_results)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Detecções auto: {self.contador_bracos_cruzados}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Gravações manuais: {self.contador_gravacoes_manuais}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
        
        # Status do buffer
        with self.buffer_lock:
            buffer_info = f"Buffer: {len(self.buffer_frames)}/{self.buffer_frames.maxlen} frames"
        cv2.putText(frame, buffer_info, (10, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        
        if self.gravando_video:
            cv2.putText(frame, "SALVANDO VIDEO...", (frame.shape[1]//2 - 150, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        return frame
    
    def cleanup(self):
        """Limpa recursos ao finalizar"""
        self.executando = False
        if LED_AVAILABLE:
            led_line.set_value(0)
            led_line.release()
            chip.close()

# -----------------------------------------------------------------------------------------------
# Callback principal do pipeline
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    """Callback principal que processa cada frame do pipeline"""
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    # Incrementa contador
    user_data.increment()
    user_data.frame_count = user_data.get_count()
    
    # Obtém informações do frame
    format, width, height = get_caps_from_pad(pad)
    
    # Obtém frame se necessário
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Adiciona ao buffer
        timestamp = time.time()
        with user_data.buffer_lock:
            user_data.buffer_frames.append(frame.copy())
            user_data.buffer_timestamps.append(timestamp)
        
        user_data.current_frame = frame
    
    # Processa detecções com intervalo
    current_time = time.time()
    if current_time - user_data.last_detection_time >= DETECTION_INTERVAL:
        user_data.last_detection_time = current_time
        
        # Obtém detecções
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        
        user_data.detection_results = []
        algum_braco_cruzado = False
        
        # Processa cada pessoa detectada
        for detection in detections:
            if detection.get_label() == "person":
                bbox = detection.get_bbox()
                landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
                
                if len(landmarks) > 0:
                    # Detecta braços cruzados
                    bracos_cruzados, debug_info = user_data.detectar_bracos_cruzados(
                        landmarks, bbox, width, height
                    )
                    
                    if bracos_cruzados:
                        algum_braco_cruzado = True
                    
                    # Coleta keypoints para visualização
                    keypoints_dict = {}
                    points = landmarks[0].get_points()
                    for name, idx in KEYPOINTS.items():
                        if idx < len(points):
                            point = points[idx]
                            x = int((point.x() * bbox.width() + bbox.xmin()) * width)
                            y = int((point.y() * bbox.height() + bbox.ymin()) * height)
                            keypoints_dict[name] = (x, y)
                    
                    user_data.detection_results.append({
                        'bbox': bbox,
                        'keypoints': keypoints_dict,
                        'bracos_cruzados': bracos_cruzados,
                        'debug_info': debug_info
                    })
        
        # Atualiza estado de gravação
        if algum_braco_cruzado:
            if user_data.bracos_cruzados_start_time is None:
                user_data.bracos_cruzados_start_time = current_time
            elif current_time - user_data.bracos_cruzados_start_time >= BRACOS_CRUZADOS_THRESHOLD:
                user_data.grava = True
                if not user_data.ultimo_estado_gravacao:
                    user_data.contador_bracos_cruzados += 1
                    user_data.pulse_led(3.0)
                    user_data.ultimo_estado_gravacao = True
        else:
            user_data.bracos_cruzados_start_time = None
            user_data.grava = False
            user_data.ultimo_estado_gravacao = False
    
    # Aplica visualização se tiver frame
    if frame is not None:
        frame = user_data.draw_visualization(frame)
        user_data.set_frame(frame)
        
        # Mostra frame em janela (opcional - comentar para headless)
        cv2.imshow("Detecção de Braços Cruzados - Hailo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            user_data.executando = False
    
    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Cria instância da aplicação
    user_data = HailoArmsDetectionApp()
    
    print("=== Sistema de Detecção de Braços Cruzados com Hailo ===")
    print(f"Buffer: {BUFFER_TAMANHO_SEGUNDOS} segundos")
    print(f"Detecção: {1/DETECTION_INTERVAL:.1f} vezes por segundo")
    print(f"Pasta de gravação: {PASTA_GRAVACAO}")
    print("\nIniciando pipeline...")
    
    try:
        # Cria e executa aplicação
        app = GStreamerPoseEstimationApp(app_callback, user_data)
        app.run()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")
    finally:
        # Limpa recursos
        user_data.cleanup()
        cv2.destroyAllWindows()
        print("Aplicação finalizada.")