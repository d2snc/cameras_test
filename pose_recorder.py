#!/usr/bin/env python3

import argparse
import cv2
import time
from datetime import datetime

# Importa a biblioteca gpiozero, recomendada para o Raspberry Pi 5
from gpiozero import LED

# Importa as funções de pós-processamento para o YOLOv8 Pose
from pose_utils import postproc_yolov8_pose

# Importa as bibliotecas da Picamera2 e do Hailo
from picamera2 import MappedArray, Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput
from picamera2.devices import Hailo

# --- Configuração Inicial ---

# Configura o pino do LED usando gpiozero.
led = LED(17)

# Analisador de argumentos para especificar o caminho do modelo
parser = argparse.ArgumentParser(description='Detecção de Pose com Gravação e LED no Hailo (Compatível com Pi 5)')
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

# --- PARÂMETROS DE DETECÇÃO AJUSTÁVEIS ---
# Número de frames consecutivos que a pose deve ser detectada para acionar a gravação.
# Aumente este valor para exigir que a pose seja mantida por mais tempo.
POSE_TRIGGER_FRAMES = 5 
pose_detected_counter = 0 # Contador para a estabilização da pose

# Variáveis globais
last_predictions = None
recording = False

# --- Funções Principais ---

def check_arms_crossed_above_head(keypoints, joint_scores, threshold=0.6):
    """
    LÓGICA REFINADA E FINAL: Verifica se os braços estão cruzados acima da cabeça
    usando o nariz como ponto de referência central.
    """
    # Pontos-chave essenciais para esta verificação precisa
    required_indices = [L_WRIST, R_WRIST, NOSE]
    
    # 1. Garante que os pontos-chave essenciais tenham alta confiança
    if not all(joint_scores[i] > threshold for i in required_indices):
        return False

    # 2. Extrai as coordenadas necessárias
    left_wrist_x, left_wrist_y = keypoints[L_WRIST]
    right_wrist_x, right_wrist_y = keypoints[R_WRIST]
    nose_x, nose_y = keypoints[NOSE]

    # 3. Condição de Altura: Ambos os pulsos devem estar ACIMA do nariz
    arms_are_up = (left_wrist_y < nose_y) and (right_wrist_y < nose_y)

    # 4. Condição de Cruzamento: Pulsos devem ter cruzado a linha vertical do nariz
    arms_are_crossed = (left_wrist_x > nose_x) and (right_wrist_x < nose_x)

    # 5. Retorna verdadeiro apenas se AMBAS as condições forem satisfeitas
    return arms_are_up and arms_are_crossed

# (As funções visualize_pose_estimation_result e draw_predictions permanecem inalteradas)
def visualize_pose_estimation_result(results, image, model_size, detection_threshold=0.5, joint_threshold=0.5):
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

            pose_found_this_frame = False
            if last_predictions and not recording:
                scores, keypoints, joint_scores = (
                    last_predictions['scores'][0], last_predictions['keypoints'][0], last_predictions['joint_scores'][0])
                for i in range(len(scores)):
                    if scores[i][0] > 0.5 and check_arms_crossed_above_head(keypoints[i], joint_scores[i].flatten()):
                        pose_found_this_frame = True
                        break
            
            # --- LÓGICA DE ESTABILIZAÇÃO (DEBOUNCE) ---
            if pose_found_this_frame:
                pose_detected_counter += 1 # Incrementa o contador se a pose for encontrada
            else:
                pose_detected_counter = 0 # Zera o contador se a pose for perdida

            # Verifica se o contador atingiu o limite para acionar a gravação
            if pose_detected_counter >= POSE_TRIGGER_FRAMES and not recording:
                recording = True
                print(f"✅ Pose confirmada por {POSE_TRIGGER_FRAMES} frames! Acionando LED e gravação...")
                
                led.on()
                time.sleep(3)
                led.off()

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"gravacao_{timestamp}.h264"
                circular_output.fileoutput = filename
                circular_output.start()
                circular_output.stop()

                print(f"✅ Gravação '{filename}' finalizada. Sistema pronto para nova detecção.")
                
                # Zera o contador e libera a gravação para a próxima detecção
                pose_detected_counter = 0
                recording = False

finally:
    if picam2.is_open:
        picam2.stop_recording()
    print("\nPrograma encerrado.")