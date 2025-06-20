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

def blink_led(times, duration):
    """Pisca o LED um número específico de vezes por uma duração"""
    for _ in range(times):
        led.on()
        time.sleep(duration)
        led.off()
        time.sleep(duration)

def continuous_error_blink():
    """Pisca o LED continuamente a cada 0.5 segundos quando há erro crítico"""
    print("❌ Erro crítico detectado! LED piscando continuamente...")
    try:
        while True:
            led.on()
            time.sleep(0.5)
            led.off()
            time.sleep(0.5)
    except KeyboardInterrupt:
        led.off()
        print("\nPrograma interrompido pelo usuário.")
    except:
        led.off()

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

def trim_video_to_last_60_seconds(video_path):
    """Corta o vídeo para manter apenas os últimos 60 segundos"""
    try:
        # Primeiro, obtém a duração total do vídeo
        duration_command = f"ffprobe -v quiet -show_entries format=duration -of csv=p=0 {video_path}"
        result = subprocess.run(duration_command, shell=True, capture_output=True, text=True, check=True)
        total_duration = float(result.stdout.strip())
        
        # Se o vídeo tem mais de 60 segundos, corta os últimos 60
        if total_duration > 60:
            start_time = total_duration - 60
            temp_video_path = video_path.replace('.mp4', '_temp.mp4')
            
            # Comando para cortar os últimos 60 segundos
            trim_command = f"ffmpeg -i {video_path} -ss {start_time} -t 60 -c copy {temp_video_path}"
            subprocess.run(trim_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Substitui o arquivo original
            os.replace(temp_video_path, video_path)
            print(f"-> Vídeo cortado para os últimos 60 segundos.")
        else:
            print(f"-> Vídeo tem {total_duration:.1f}s, mantendo duração original.")
            
    except Exception as e:
        print(f"❌ Erro ao cortar vídeo: {e}")

def main_program():
    """Função principal do programa"""
    global pose_detected_counter, last_predictions, recording, model_size
    
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
            
            # Pisca o LED 3 vezes por 0.5 segundos para indicar que o sistema está pronto
            blink_led(3, 0.5)
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
                    
                    # Acende o LED por 1 segundo quando começar a gravar
                    led.on()
                    time.sleep(1)
                    led.off()
                    
                    base_filename = datetime.now().strftime("gravacao_%Y-%m-%d_%H-%M-%S")
                    h264_filename = base_filename + ".h264"
                    mp4_filename = base_filename + ".mp4"
                    circular_output.fileoutput = h264_filename
                    circular_output.start()
                    circular_output.stop()
                    print(f"-> Arquivo temporário '{h264_filename}' salvo.")
                    print(f"-> Convertendo para '{mp4_filename}' com FFmpeg...")
                    
                    conversion_success = False
                    try:
                        # Conversão para MP4
                        command = f"ffmpeg -framerate 30 -i {h264_filename} -c:v copy {mp4_filename}"
                        subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        print(f"✅ Gravação convertida para '{mp4_filename}'.")
                        
                        # Corta o vídeo para os últimos 60 segundos
                        trim_video_to_last_60_seconds(mp4_filename)
                        
                        print(f"✅ Gravação final salva como '{mp4_filename}'.")
                        conversion_success = True
                        
                        # Pisca 2 vezes por 0.5 segundos quando a gravação for concluída com sucesso
                        blink_led(2, 0.5)
                        
                    except subprocess.CalledProcessError as e:
                        print(f"❌ Erro ao converter com FFmpeg: {e}")
                        # Pisca 6 vezes por 0.3 segundos quando a gravação falhar
                        blink_led(6, 0.3)
                    finally:
                        if os.path.exists(h264_filename):
                            os.remove(h264_filename)
                            print(f"-> Arquivo temporário '{h264_filename}' removido.")
                    
                    print("-> Sistema pronto para nova detecção.")
                    pose_detected_counter = 0
                    recording = False
    
    except KeyboardInterrupt:
        print("\nPrograma interrompido pelo usuário.")
        if picam2.is_open:
            picam2.stop_recording()
        led.off()
        return  # Sai normalmente sem erro
        
    except Exception as e:
        print(f"❌ Erro no programa principal: {e}")
        try:
            if picam2.is_open:
                picam2.stop_recording()
        except:
            pass
        raise  # Re-levanta a exceção para ser capturada pelo handler principal
    
    finally:
        try:
            if picam2.is_open:
                picam2.stop_recording()
        except:
            pass

# Execução principal com handler de erro
if __name__ == "__main__":
    try:
        main_program()
        print("\nPrograma encerrado normalmente.")
    except KeyboardInterrupt:
        print("\nPrograma interrompido pelo usuário.")
        led.off()
    except Exception as e:
        print(f"❌ Erro crítico: {e}")
        # Em vez de encerrar, pisca o LED continuamente
        continuous_error_blink()
