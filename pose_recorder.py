#!/usr/bin/env python3
import argparse
import cv2
import time
from datetime import datetime
import subprocess
import os
from gpiozero import LED
from pose_utils import postproc_yolov8_pose
from picamera2 import MappedArray, Picamera2 #, Preview
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

POSE_TRIGGER_FRAMES = 10
pose_detected_counter = 0
last_predictions = None
recording = False

def check_arms_crossed_above_head(keypoints, joint_scores, threshold=0.6):
    required_indices = [L_WRIST, R_WRIST, NOSE]
    if not all(joint_scores[i] > threshold for i in required_indices):
        return False
    left_wrist_x, left_wrist_y = keypoints[L_WRIST]
    right_wrist_x, right_wrist_y = keypoints[R_WRIST]
    nose_x, nose_y = keypoints[NOSE]
    arms_are_up = (left_wrist_y < nose_y) and (right_wrist_y < nose_y)
    arms_are_crossed = (left_wrist_x > nose_x) and (right_wrist_x < nose_x)
    return arms_are_up and arms_are_crossed

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
        
        # Configuração do buffer - usar buffer grande para garantir pelo menos 60 segundos
        fps = 30
        target_duration_seconds = 60
        bitrate = 2000000
        encoder = H264Encoder(bitrate=bitrate)
        
        # Buffer generoso para garantir que sempre tenha mais de 60 segundos
        buffer_size_bytes = 150 * 1024 * 1024  # 150 MB - suficiente para ~90+ segundos
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
                
                
                base_filename = datetime.now().strftime("gravacao_%Y-%m-%d_%H-%M-%S")
                h264_filename = base_filename + ".h264"
                h264_temp_filename = base_filename + "_temp.h264"
                mp4_filename = base_filename + ".mp4"
                
                # Salvar o buffer circular completo primeiro
                with open(h264_temp_filename, "wb") as f:
                    circular_output.fileoutput = f
                    circular_output.copy_to_file()
                
                print(f"-> Buffer completo salvo como '{h264_temp_filename}'.")
                print(f"-> Cortando para exatos {target_duration_seconds} segundos...")
                
                try:
                    # PASSO 1: Cortar para exatos 60 segundos (pegar os últimos 60 segundos)
                    # -ss: pular os primeiros X segundos (será calculado dinamicamente)
                    # -t: duração exata de 60 segundos
                    # -c:v copy: cópia sem recodificação
                    
                    # Primeiro, descobrir a duração total do vídeo
                    duration_command = f"ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {h264_temp_filename}"
                    duration_result = subprocess.run(duration_command, shell=True, capture_output=True, text=True)
                    
                    if duration_result.returncode == 0:
                        total_duration = float(duration_result.stdout.strip())
                        print(f"-> Duração total do buffer: {total_duration:.1f} segundos")
                        
                        if total_duration > target_duration_seconds:
                            # Calcular ponto de início para pegar os últimos 60 segundos
                            start_time = total_duration - target_duration_seconds
                            trim_command = f"ffmpeg -ss {start_time} -i {h264_temp_filename} -t {target_duration_seconds} -c:v copy {h264_filename}"
                        else:
                            # Se o buffer tem menos de 60 segundos, usar tudo
                            trim_command = f"ffmpeg -i {h264_temp_filename} -c:v copy {h264_filename}"
                            print(f"-> Aviso: Buffer tem apenas {total_duration:.1f} segundos")
                    else:
                        # Se não conseguir determinar duração, usar comando básico
                        trim_command = f"ffmpeg -i {h264_temp_filename} -t {target_duration_seconds} -c:v copy {h264_filename}"
                    
                    subprocess.run(trim_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"-> Vídeo cortado salvo como '{h264_filename}'.")
                    
                    # PASSO 2: Converter para MP4
                    convert_command = f"ffmpeg -framerate {fps} -i {h264_filename} -c:v copy {mp4_filename}"
                    subprocess.run(convert_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"✅ Gravação final de {target_duration_seconds} segundos salva como '{mp4_filename}'.")
                    
                except subprocess.CalledProcessError as e:
                    print(f"❌ Erro ao processar vídeo: {e}")
                finally:
                    # Limpar arquivos temporários
                    for temp_file in [h264_temp_filename, h264_filename]:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            print(f"-> Arquivo temporário '{temp_file}' removido.")
                
                print("-> Sistema pronto para nova detecção.")
                pose_detected_counter = 0
                print(f"✅ Pose confirmada por {POSE_TRIGGER_FRAMES} frames! Acionando LED e gravação...")
                led.on()
                time.sleep(3)
                led.off()
                recording = False

finally:
    if picam2.is_open:
        picam2.stop_recording()
    print("\nPrograma encerrado.")
    led.on()
    time.sleep(10)
    led.off()
