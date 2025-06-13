import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
from collections import deque
import threading
import time
from datetime import datetime

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.use_frame = True  # Sempre usar frame para salvar vídeo
        self.fps = 30  # Assumindo 30 FPS
        self.buffer_seconds = 30
        self.buffer_size = self.fps * self.buffer_seconds
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.is_recording = False
        self.crossed_arms_detected = False
        self.crossed_arms_message = ""
        self.message_timer = 0
        self.last_save_time = 0
        self.save_cooldown = 5  # Segundos entre salvamentos
        self.video_writer = None
        self.recording_thread = None
        
    def add_frame_to_buffer(self, frame):
        """Adiciona frame ao buffer circular"""
        if frame is not None:
            self.frame_buffer.append(frame.copy())
    
    def save_buffer_to_video(self, filename):
        """Salva o buffer de frames em um arquivo de vídeo"""
        if len(self.frame_buffer) == 0:
            return
        
        # Criar uma cópia do buffer para evitar problemas de concorrência
        frames_to_save = list(self.frame_buffer)
        
        # Configurar o VideoWriter
        height, width = frames_to_save[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))
        
        # Escrever frames
        for frame in frames_to_save:
            out.write(frame)
        
        out.release()
        print(f"Vídeo salvo: {filename}")
    
    def save_video_thread(self):
        """Thread para salvar vídeo sem bloquear o pipeline"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crossed_arms_{timestamp}.mp4"
        self.save_buffer_to_video(filename)
        self.is_recording = False

# -----------------------------------------------------------------------------------------------
# Funções auxiliares
# -----------------------------------------------------------------------------------------------

def check_arms_crossed(points, keypoints, bbox, width, height):
    """
    Verifica se os braços estão cruzados acima da cabeça
    """
    try:
        # Obter índices dos pontos chave
        left_wrist_idx = keypoints['left_wrist']
        right_wrist_idx = keypoints['right_wrist']
        left_elbow_idx = keypoints['left_elbow']
        right_elbow_idx = keypoints['right_elbow']
        left_shoulder_idx = keypoints['left_shoulder']
        right_shoulder_idx = keypoints['right_shoulder']
        nose_idx = keypoints['nose']
        left_eye_idx = keypoints['left_eye']
        right_eye_idx = keypoints['right_eye']
        
        # Converter coordenadas para pixels
        def get_pixel_coords(point_idx):
            point = points[point_idx]
            x = int((point.x() * bbox.width() + bbox.xmin()) * width)
            y = int((point.y() * bbox.height() + bbox.ymin()) * height)
            return x, y
        
        # Obter coordenadas
        left_wrist = get_pixel_coords(left_wrist_idx)
        right_wrist = get_pixel_coords(right_wrist_idx)
        left_elbow = get_pixel_coords(left_elbow_idx)
        right_elbow = get_pixel_coords(right_elbow_idx)
        left_shoulder = get_pixel_coords(left_shoulder_idx)
        right_shoulder = get_pixel_coords(right_shoulder_idx)
        nose = get_pixel_coords(nose_idx)
        left_eye = get_pixel_coords(left_eye_idx)
        right_eye = get_pixel_coords(right_eye_idx)
        
        # Calcular altura da cabeça (usar o ponto mais alto entre nariz e olhos)
        head_y = min(nose[1], left_eye[1], right_eye[1])
        
        # Verificar se os pontos são válidos
        if all(coord[0] > 0 and coord[1] > 0 for coord in [left_wrist, right_wrist, left_elbow, right_elbow]):
            # Critérios para braços cruzados acima da cabeça:
            # 1. Pulso esquerdo está à direita do centro do corpo
            # 2. Pulso direito está à esquerda do centro do corpo
            # 3. Ambos os pulsos estão acima da cabeça
            # 4. Pulsos cruzados (pulso esquerdo à direita do direito ou vice-versa)
            
            body_center_x = (left_shoulder[0] + right_shoulder[0]) // 2
            
            # Verificar se os braços estão cruzados
            wrists_crossed = (left_wrist[0] > right_wrist[0])  # Pulso esquerdo à direita do direito
            
            # Verificar se ambos os pulsos estão acima da cabeça
            wrists_above_head = (left_wrist[1] < head_y - 10) and (right_wrist[1] < head_y - 10)
            
            # Verificar se os cotovelos também estão elevados
            elbows_elevated = (left_elbow[1] < nose[1] + 50) and (right_elbow[1] < nose[1] + 50)
            
            # Verificar se os pulsos estão próximos um do outro (indicando cruzamento)
            wrists_close = abs(left_wrist[0] - right_wrist[0]) < 100
            
            return wrists_crossed and wrists_above_head and elbows_elevated and wrists_close
            
    except Exception as e:
        print(f"Erro ao verificar braços cruzados: {e}")
        return False
    
    return False

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # Get video frame
    frame = None
    if format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Get the keypoints
    keypoints = get_keypoints()

    # Flag para detectar se alguma pessoa cruzou os braços
    arms_crossed_in_frame = False

    # Parse the detections
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        if label == "person":
            # Get track ID
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
            string_to_print += (f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n")

            # Pose estimation landmarks from detection
            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if len(landmarks) != 0:
                points = landmarks[0].get_points()
                
                # Verificar se os braços estão cruzados
                if check_arms_crossed(points, keypoints, bbox, width, height):
                    arms_crossed_in_frame = True
                    string_to_print += f"BRAÇOS CRUZADOS ACIMA DA CABEÇA - ID: {track_id}\n"
                
                # Desenhar pontos chave (opcional)
                for keypoint_name, keypoint_index in keypoints.items():
                    if keypoint_index < len(points):
                        point = points[keypoint_index]
                        x = int((point.x() * bbox.width() + bbox.xmin()) * width)
                        y = int((point.y() * bbox.height() + bbox.ymin()) * height)
                        if frame is not None:
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Processar detecção de braços cruzados
    current_time = time.time()
    
    if arms_crossed_in_frame:
        if not user_data.crossed_arms_detected:
            user_data.crossed_arms_detected = True
            user_data.crossed_arms_message = "BRAÇOS CRUZADOS ACIMA!"
            user_data.message_timer = current_time + 3  # Mostrar mensagem por 3 segundos
            
            # Verificar se podemos salvar (cooldown)
            if current_time - user_data.last_save_time > user_data.save_cooldown:
                if not user_data.is_recording and len(user_data.frame_buffer) >= user_data.buffer_size:
                    user_data.is_recording = True
                    user_data.last_save_time = current_time
                    # Iniciar thread para salvar vídeo
                    user_data.recording_thread = threading.Thread(target=user_data.save_video_thread)
                    user_data.recording_thread.start()
    else:
        user_data.crossed_arms_detected = False

    # Adicionar frame ao buffer
    if frame is not None:
        # Criar cópia do frame para o buffer
        buffer_frame = frame.copy()
        
        # Mostrar mensagem na tela se necessário
        if user_data.message_timer > current_time:
            # Adicionar texto grande e verde
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3
            thickness = 5
            text = user_data.crossed_arms_message
            
            # Obter dimensões do texto
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Posição centralizada
            text_x = (width - text_width) // 2
            text_y = height // 2
            
            # Adicionar fundo preto para melhor visibilidade
            cv2.rectangle(frame, 
                         (text_x - 10, text_y - text_height - 10),
                         (text_x + text_width + 10, text_y + baseline + 10),
                         (0, 0, 0), -1)
            
            # Adicionar texto verde
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
        
        # Adicionar indicador de gravação
        if user_data.is_recording:
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "GRAVANDO", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Adicionar frame ao buffer
        user_data.add_frame_to_buffer(buffer_frame)
        
        # Definir frame para exibição
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

# This function can be used to get the COCO keypoints coorespondence map
def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    keypoints = {
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
    return keypoints

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()