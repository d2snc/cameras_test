#!/usr/bin/env python3

import argparse
import cv2
import time
from datetime import datetime

# Importa a biblioteca para controle dos pinos GPIO
import RPi.GPIO as GPIO

# Importa as funções de pós-processamento para o YOLOv8 Pose
from pose_utils import postproc_yolov8_pose

# Importa as bibliotecas da Picamera2 e do Hailo
from picamera2 import MappedArray, Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput
from picamera2.devices import Hailo

# --- Configuração Inicial ---

# Define o pino do LED
LED_PIN = 17

# Configuração do GPIO
GPIO.setmode(GPIO.BCM)  # Usa a numeração de pinos BCM
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW) # Garante que o LED comece desligado

# Analisador de argumentos para especificar o caminho do modelo
parser = argparse.ArgumentParser(description='Detecção de Pose com Gravação e LED no Hailo')
parser.add_argument('-m', '--model', help="Caminho para o arquivo .hef", default="/usr/share/hailo-models/yolov8s_pose_h8l_pi.hef")
args = parser.parse_args()

# Índices dos pontos-chave (keypoints) do corpo
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, \
    L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = range(17)

# Pares de articulações para desenhar o esqueleto
JOINT_PAIRS = [
    [L_SHOULDER, R_SHOULDER], [L_SHOULDER, L_ELBOW], [L_ELBOW, L_WRIST],
    [R_SHOULDER, R_ELBOW], [R_ELBOW, R_WRIST], [L_SHOULDER, L_HIP],
    [R_SHOULDER, R_HIP], [L_HIP, R_HIP]
]

# Variáveis globais
last_predictions = None
recording = False

# --- Funções Principais ---

def check_arms_crossed_above_head(keypoints, joint_scores, threshold=0.5):
    """Verifica se a pose 'braços cruzados acima da cabeça' foi detectada."""
    if all(joint_scores[i] > threshold for i in [L_WRIST, R_WRIST, L_SHOULDER, R_SHOULDER, NOSE]):
        left_wrist_y, right_wrist_y = keypoints[L_WRIST][1], keypoints[R_WRIST][1]
        left_shoulder_y, right_shoulder_y = keypoints[L_SHOULDER][1], keypoints[R_SHOULDER][1]
        nose_y = keypoints[NOSE][1]
        if left_wrist_y < nose_y and right_wrist_y < nose_y and \
           left_wrist_y < (left_shoulder_y + right_shoulder_y) / 2 and \
           right_wrist_y < (left_shoulder_y + right_shoulder_y) / 2:
            return True
    return False

def visualize_pose_estimation_result(results, image, model_size, detection_threshold=0.5, joint_threshold=0.5):
    """Desenha os resultados da detecção de pose na imagem."""
    image_size = (image.shape[1], image.shape[0])
    def scale_coord(coord):
        return tuple([int(c * t / f) for c, f, t in zip(coord, model_size, image_size)])
    if not results or 'bboxes' not in results:
        return
    bboxes, scores, keypoints, joint_scores = (
        results['bboxes'][0], results['scores'][0], results['keypoints'][0], results['joint_scores'][0])
    for detection_box, detection_score, detection_keypoints, detection_keypoints_score in \
            zip(bboxes, scores, keypoints, joint_scores):
        if detection_score[0] < detection_threshold:
            continue
        coord_min, coord_max = scale_coord(detection_box[:2]), scale_coord(detection_box[2:])
        cv2.rectangle(image, coord_min, coord_max, (0, 255, 0), 2)
        joint_visible = detection_keypoints_score.flatten() > joint_threshold
        for joint0_idx, joint1_idx in JOINT_PAIRS:
            if joint_visible[joint0_idx] and joint_visible[joint1_idx]:
                p1, p2 = scale_coord(detection_keypoints[joint0_idx]), scale_coord(detection_keypoints[joint1_idx])
                cv2.line(image, p1, p2, (255, 0, 255), 3)

def draw_predictions(request):
    """Callback para desenhar as predições na janela de preview."""
    with MappedArray(request, 'main') as m:
        if last_predictions:
            visualize_pose_estimation_result(last_predictions, m.array, model_size)

# --- Lógica Principal de Execução ---

picam2 = Picamera2()
try:
    with Hailo(args.model) as hailo:
        main_size = (1280, 720)
        model_h, model_w, _ = hailo.get_input_shape()
        model_size = lores_size = (model_w, model_h)

        config = picam2.create_video_configuration(
            main={'size': main_size, 'format': 'XRGB8888'},
            lores={'size': lores_size, 'format': 'RGB888'},
            controls={'FrameRate': 30}
        )
        picam2.configure(config)
        picam2.start_preview(Preview.QTGL, x=0, y=0, width=main_size[0] // 2, height=main_size[1] // 2)
        
        bitrate = 10000000
        encoder = H264Encoder(bitrate=bitrate)
        
        seconds_to_buffer = 20
        buffer_size_bytes = int(bitrate / 8 * seconds_to_buffer)
        circular_output = CircularOutput(buffersize=buffer_size_bytes)
        
        picam2.start_recording(encoder, circular_output)
        picam2.pre_callback = draw_predictions
        
        print("🚀 Sistema iniciado. Aguardando detecção da pose...")

        while True:
            frame = picam2.capture_array('lores')
            raw_detections = hailo.run(frame)
            last_predictions = postproc_yolov8_pose(1, raw_detections, model_size)

            pose_found = False
            if last_predictions and not recording:
                scores, keypoints, joint_scores = (
                    last_predictions['scores'][0], last_predictions['keypoints'][0], last_predictions['joint_scores'][0])
                for i in range(len(scores)):
                    if scores[i][0] > 0.5 and check_arms_crossed_above_head(keypoints[i], joint_scores[i].flatten()):
                        pose_found = True
                        break
            
            if pose_found and not recording:
                recording = True
                
                # --- CONTROLE DO LED E GRAVAÇÃO ---
                print(f"✅ Pose detectada! Acionando LED e gravação...")
                
                # Liga o LED por 3 segundos
                GPIO.output(LED_PIN, GPIO.HIGH)
                time.sleep(3)
                GPIO.output(LED_PIN, GPIO.LOW)

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"gravacao_{timestamp}.h264"
                circular_output.fileoutput = filename
                circular_output.start()
                circular_output.stop()

                print(f"✅ Gravação '{filename}' finalizada. Sistema pronto para nova detecção.")
                recording = False

finally:
    # Garante que tudo seja desligado e limpo corretamente na saída
    if picam2.is_open:
        picam2.stop_recording()
    GPIO.cleanup() # Limpa as configurações dos pinos GPIO
    print("\nPrograma encerrado. Pinos GPIO limpos.")