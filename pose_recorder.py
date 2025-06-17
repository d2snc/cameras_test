#!/usr/bin/env python3

import argparse
import cv2
import time
from datetime import datetime
import subprocess
import os
from gpiozero import LED
from pose_utils import postproc_yolov8_pose
from picamera2 import MappedArray, Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput
from picamera2.devices import Hailo

led = LED(17)
parser = argparse.ArgumentParser(description='Detecção de Pose com FFmpeg (Compatível com Pi 5)')
parser.add_argument('-m', '--model', help="Caminho para o arquivo .hef", default="/usr/share/hailo-models/yolov8s_pose_h8l_pi.hef")
args = parser.parse_args()

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, \
    L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = range(17)

JOINT_PAIRS = [[L_SHOULDER, R_SHOULDER], [L_SHOULDER, L_ELBOW], [L_ELBOW, L_WRIST], [R_SHOULDER, R_ELBOW], [R_ELBOW, R_WRIST], [L_SHOULDER, L_HIP], [R_SHOULDER, R_HIP], [L_HIP, R_HIP]]

POSE_TRIGGER_FRAMES = 5 
pose_detected_counter = 0
last_predictions = None
recording = False

def check_arms_crossed_above_head(keypoints, joint_scores, threshold=0.7):  # Aumentei o threshold
    # Índices necessários incluindo ombros para melhor detecção
    required_indices = [L_WRIST, R_WRIST, NOSE, L_SHOULDER, R_SHOULDER]
    
    # Verifica se todos os pontos necessários têm confiança suficiente
    if not all(joint_scores[i] > threshold for i in required_indices):
        return False
    
    # Extrai coordenadas
    left_wrist_x, left_wrist_y = keypoints[L_WRIST]
    right_wrist_x, right_wrist_y = keypoints[R_WRIST]
    nose_x, nose_y = keypoints[NOSE]
    left_shoulder_x, left_shoulder_y = keypoints[L_SHOULDER]
    right_shoulder_x, right_shoulder_y = keypoints[R_SHOULDER]
    
    # 1. Verifica se AMBOS os pulsos estão bem acima do nariz (não apenas acima)
    # Adiciona margem de segurança para garantir que os braços estão realmente levantados
    margin_above_head = nose_y * 0.2  # 20% da altura do nariz como margem
    arms_are_up = (left_wrist_y < (nose_y - margin_above_head)) and \
                  (right_wrist_y < (nose_y - margin_above_head))
    
    # 2. Verifica se os braços estão REALMENTE cruzados
    # O pulso esquerdo deve estar do lado direito do corpo E
    # o pulso direito deve estar do lado esquerdo do corpo
    body_center_x = (left_shoulder_x + right_shoulder_x) / 2
    
    # Usa o centro do corpo como referência em vez do nariz
    left_arm_crossed_right = left_wrist_x > body_center_x
    right_arm_crossed_left = right_wrist_x < body_center_x
    
    arms_are_crossed = left_arm_crossed_right and right_arm_crossed_left
    
    # 3. Verifica se os pulsos estão próximos um do outro (característica de cruzamento)
    wrist_distance = abs(left_wrist_x - right_wrist_x)
    shoulder_width = abs(left_shoulder_x - right_shoulder_x)
    
    # Os pulsos devem estar mais próximos que 50% da largura dos ombros
    wrists_are_close = wrist_distance < (shoulder_width * 0.5)
    
    # 4. Verifica se os braços estão acima dos ombros
    arms_above_shoulders = (left_wrist_y < left_shoulder_y) and \
                          (right_wrist_y < right_shoulder_y)
    
    # Retorna True apenas se TODAS as condições forem atendidas
    return arms_are_up and arms_are_crossed and wrists_are_close and arms_above_shoulders

def visualize_pose_estimation_result(results, image, model_size, detection_threshold=0.5, joint_threshold=0.5):
    image_size = (image.shape[1], image.shape[0])
    def scale_coord(coord): return tuple([int(c * t / f) for c, f, t in zip(coord, model_size, image_size)])
    if not results or 'bboxes' not in results: return
    bboxes, scores, keypoints, joint_scores = (results['bboxes'][0], results['scores'][0], results['keypoints'][0], results['joint_scores'][0])
    for detection_box, detection_score, detection_keypoints, detection_keypoints_score in zip(bboxes, scores, keypoints, joint_scores):
        if detection_score[0] < detection_threshold: continue
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

picam2 = Picamera2()
try:
    with Hailo(args.model) as hailo:
        main_size = (1280, 720)
        model_h, model_w, _ = hailo.get_input_shape()
        model_size = lores_size = (model_w, model_h)
        config = picam2.create_video_configuration(main={'size': main_size, 'format': 'XRGB8888'}, lores={'size': lores_size, 'format': 'RGB888'}, controls={'FrameRate': 30})
        picam2.configure(config)
        #picam2.start_preview(Preview.QTGL, x=0, y=0, width=main_size[0] // 2, height=main_size[1] // 2)

        bitrate = 10000000
        encoder = H264Encoder(bitrate=bitrate)
        seconds_to_buffer = 20
        buffer_size_bytes = int(bitrate / 8 * seconds_to_buffer)
        circular_output = CircularOutput(buffersize=buffer_size_bytes)

        picam2.start_recording(encoder, circular_output)
        #Descomentar abaixo se quiser visualizar as predições na tela
        #picam2.pre_callback = draw_predictions

        # pisca o led 3 vezes para indicar que o sistema está pronto
        for _ in range(3):
            led.on()
            time.sleep(1)
            led.off()

        print("🚀 Sistema iniciado. Aguardando detecção da pose...")

        while True:
            frame = picam2.capture_array('lores')
            raw_detections = hailo.run(frame)
            last_predictions = postproc_yolov8_pose(1, raw_detections, model_size)

            pose_found_this_frame = False
            if last_predictions and not recording:
                scores, keypoints, joint_scores = (last_predictions['scores'][0], last_predictions['keypoints'][0], last_predictions['joint_scores'][0])
                for i in range(len(scores)):
                    if scores[i][0] > 0.5 and check_arms_crossed_above_head(keypoints[i], joint_scores[i].flatten()):
                        pose_found_this_frame = True
                        break

            if pose_found_this_frame:
                pose_detected_counter += 1
            else:
                pose_detected_counter = 0

            if pose_detected_counter >= POSE_TRIGGER_FRAMES and not recording:
                recording = True
                print(f"✅ Pose confirmada por {POSE_TRIGGER_FRAMES} frames! Acionando LED e gravação...")

                led.on()
                time.sleep(3)
                led.off()

                base_filename = datetime.now().strftime("gravacao_%Y-%m-%d_%H-%M-%S")
                h264_filename = base_filename + ".h264"
                mp4_filename = base_filename + ".mp4"

                circular_output.fileoutput = h264_filename
                circular_output.start()
                circular_output.stop()
                print(f"-> Arquivo temporário '{h264_filename}' salvo.")

                print(f"-> Convertendo para '{mp4_filename}' com FFmpeg...")
                try:
                    # *** COMANDO ALTERADO PARA USAR FFmpeg ***
                    # -i: arquivo de entrada
                    # -c:v copy: copia o stream de vídeo sem recodificar (muito rápido)
                    command = f"ffmpeg -framerate 30 -i {h264_filename} -c:v copy {mp4_filename}"
                    subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"✅ Gravação final salva como '{mp4_filename}'.")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Erro ao converter com FFmpeg: {e}")
                finally:
                    if os.path.exists(h264_filename):
                        os.remove(h264_filename)
                        print(f"-> Arquivo temporário '{h264_filename}' removido.")

                print("-> Sistema pronto para nova detecção.")
                pose_detected_counter = 0
                recording = False
finally:
    if picam2.is_open:
        picam2.stop_recording()
    print("\nPrograma encerrado.")