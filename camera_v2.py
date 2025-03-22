#!/usr/bin/python3
import cv2
import mediapipe as mp
from picamera2 import Picamera2, Preview
import time

# Inicializando MediaPipe Pose e as funções de desenho
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Inicializa a variável "grava"
grava = False

# Inicializa a câmera Picamera2
picam2 = Picamera2()
picam2.start_preview(Preview.QTGL)

# Cria e aplica a configuração de preview
preview_config = picam2.create_preview_configuration(main={"format": "XBGR8888", "size": (640, 480)})
picam2.configure(preview_config)
picam2.start()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        # Captura o frame da câmera
        frame = picam2.capture_array()
        if frame is None:
            break

        # Espelha o frame para uma visualização mais natural
        frame = cv2.flip(frame, 1)

        # Converte de BGRA (XBGR8888) para BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Converte a imagem de BGR para RGB para o MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processa a imagem para detectar a pose
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # Desenha os landmarks na imagem
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Obtém os landmarks e as dimensões da imagem
            landmarks = results.pose_landmarks.landmark
            height, width, _ = frame.shape
            
            # Extrai coordenadas dos pulsos e ombros (valores normalizados)
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Converte as coordenadas normalizadas para pixels
            left_wrist_coords = (int(left_wrist.x * width), int(left_wrist.y * height))
            right_wrist_coords = (int(right_wrist.x * width), int(right_wrist.y * height))
            left_shoulder_coords = (int(left_shoulder.x * width), int(left_shoulder.y * height))
            right_shoulder_coords = (int(right_shoulder.x * width), int(right_shoulder.y * height))
            
            # Condição para detectar braços cruzados
            if left_wrist_coords[0] > right_shoulder_coords[0] and right_wrist_coords[0] < left_shoulder_coords[0]:
                grava = True
            else:
                grava = False
            
            # Exibe o status da variável "grava" na tela
            cv2.putText(frame, f"Grava: {grava}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if grava else (0, 0, 255), 2)
        
        # Exibe a imagem com os landmarks e status
        cv2.imshow("Deteccao de Bracos Cruzados", frame)
        
        # Encerra o loop ao pressionar a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libera recursos
cv2.destroyAllWindows()
picam2.stop_preview()
picam2.close()

