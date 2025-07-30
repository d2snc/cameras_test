#!/usr/bin/env python3
import argparse
import cv2
import time
from datetime import datetime
import subprocess
import os
from gpiozero import LED
from pose_utils import postproc_yolov8_pose
from picamera2 import MappedArray, Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput
from picamera2.devices import Hailo
import threading
import queue

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
parser.add_argument('-m', '--model', help="Caminho para o arquivo .hef", default="/home/d2snc/Documents/cameras_test/yolov8m_pose.hef")
args = parser.parse_args()

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, \
    L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = range(17)

JOINT_PAIRS = [[L_SHOULDER, R_SHOULDER], [L_SHOULDER, L_ELBOW], [L_ELBOW, L_WRIST], [R_SHOULDER, R_ELBOW], [R_ELBOW, R_WRIST], [L_SHOULDER, L_HIP], [R_SHOULDER, R_HIP], [L_HIP, R_HIP]]

POSE_TRIGGER_FRAMES = 10
pose_detected_counter = 0
last_predictions = None
recording = False
frame_counter = 0
last_frame_time = time.time()

def check_arms_crossed_above_head(keypoints, joint_scores, threshold=0.6):
    try:
        required_indices = [L_WRIST, R_WRIST, NOSE]
        if not all(joint_scores[i] > threshold for i in required_indices):
            return False
        left_wrist_x, left_wrist_y = keypoints[L_WRIST]
        right_wrist_x, right_wrist_y = keypoints[R_WRIST]
        nose_x, nose_y = keypoints[NOSE]
        
        # Check if both wrists are above the head (nose)
        arms_are_up = (left_wrist_y < nose_y) and (right_wrist_y < nose_y)
        
        # Calculate distance between wrists
        wrist_distance = ((left_wrist_x - right_wrist_x) ** 2 + (left_wrist_y - right_wrist_y) ** 2) ** 0.5
        
        # Define close distance threshold (adjust based on your needs)
        close_distance_threshold = 50  # pixels - you may need to adjust this
        
        # Check if wrists are close to each other
        wrists_are_close = wrist_distance < close_distance_threshold
        
        return arms_are_up and wrists_are_close
    except:
        return False

def visualize_pose_estimation_result(results, image, model_size, detection_threshold=0.3, joint_threshold=0.3):
    global pose_detected_counter, recording
    try:
        image_size = (image.shape[1], image.shape[0])
        def scale_coord(coord): return tuple([int(c * t / f) for c, f, t in zip(coord, model_size, image_size)])
        
        # Add status indicators
        status_color = (0, 255, 0) if recording else (255, 255, 255)
        status_text = "RECORDING" if recording else f"POSE COUNT: {pose_detected_counter}/{POSE_TRIGGER_FRAMES}"
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        if not results or 'bboxes' not in results: return
        bboxes, scores, keypoints, joint_scores = (results['bboxes'][0], results['scores'][0], results['keypoints'][0], results['joint_scores'][0])
        
        for detection_box, detection_score, detection_keypoints, detection_keypoints_score in zip(bboxes, scores, keypoints, joint_scores):
            if detection_score[0] < detection_threshold: continue
            
            # Check if this detection has the target pose
            has_target_pose = False
            try:
                if check_arms_crossed_above_head(detection_keypoints, detection_keypoints_score.flatten()):
                    has_target_pose = True
            except:
                pass
            
            # Use different colors for target pose vs regular detection
            bbox_color = (0, 255, 255) if has_target_pose else (0, 255, 0)  # Yellow for target pose, green for regular
            joint_color = (255, 255, 0) if has_target_pose else (255, 0, 255)  # Cyan for target pose, magenta for regular
            
            coord_min, coord_max = scale_coord(detection_box[:2]), scale_coord(detection_box[2:])
            cv2.rectangle(image, coord_min, coord_max, bbox_color, 3 if has_target_pose else 2)
            
            # Add confidence score
            conf_text = f"{detection_score[0]:.2f}"
            cv2.putText(image, conf_text, (coord_min[0], coord_min[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 1)
            
            joint_visible = detection_keypoints_score.flatten() > joint_threshold
            
            # Draw skeleton
            for joint0_idx, joint1_idx in JOINT_PAIRS:
                if joint_visible[joint0_idx] and joint_visible[joint1_idx]:
                    p1, p2 = scale_coord(detection_keypoints[joint0_idx]), scale_coord(detection_keypoints[joint1_idx])
                    cv2.line(image, p1, p2, joint_color, 4 if has_target_pose else 2)
            
            # Draw keypoints
            for i, (kp, visible) in enumerate(zip(detection_keypoints, joint_visible)):
                if visible:
                    kp_scaled = scale_coord(kp)
                    radius = 6 if has_target_pose else 4
                    cv2.circle(image, kp_scaled, radius, joint_color, -1)
                    
            # Highlight specific joints for target pose
            if has_target_pose:
                for joint_idx in [L_WRIST, R_WRIST, NOSE]:
                    if joint_visible[joint_idx]:
                        kp_scaled = scale_coord(detection_keypoints[joint_idx])
                        cv2.circle(image, kp_scaled, 8, (0, 0, 255), 2)  # Red circle around key joints
                        
    except:
        pass

def draw_predictions(request):
    try:
        with MappedArray(request, 'main') as m:
            if last_predictions:
                visualize_pose_estimation_result(last_predictions, m.array, model_size, detection_threshold=0.3, joint_threshold=0.3)
    except:
        pass

def safe_subprocess_run(command, description="comando"):
    """Executa subprocess com tratamento de erro robusto"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True, timeout=30)
        return True, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, f"Timeout ao executar {description}"
    except subprocess.CalledProcessError as e:
        return False, f"Erro ao executar {description}: {e}"
    except Exception as e:
        return False, f"Erro inesperado ao executar {description}: {e}"

def trim_video_to_last_60_seconds(video_path):
    """Corta o vídeo para manter apenas os últimos 60 segundos"""
    try:
        recording = False
        success, duration_str = safe_subprocess_run(
            f"ffprobe -v quiet -show_entries format=duration -of csv=p=0 {video_path}",
            "obter duração do vídeo"
        )
        
        if not success:
            return
            
        total_duration = float(duration_str)
        
        if total_duration > 60:
            start_time = total_duration - 60
            temp_video_path = video_path.replace('.mp4', '_temp.mp4')
            
            success, _ = safe_subprocess_run(
                f"ffmpeg -i {video_path} -ss {start_time} -t 60 -c copy {temp_video_path}",
                "cortar vídeo"
            )
            
            if success and os.path.exists(temp_video_path):
                try:
                    os.replace(temp_video_path, video_path)
                except:
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
    except:
        pass

def process_video_async(h264_filename, mp4_filename):
    """Processa vídeo em thread separada para não bloquear detecção"""
    def process():
        try:
            success, _ = safe_subprocess_run(
                f"ffmpeg -framerate 30 -i {h264_filename} -c:v copy {mp4_filename}",
                "conversão para MP4"
            )
            
            if success:
                trim_video_to_last_60_seconds(mp4_filename)
                blink_led(2, 0.5)
            else:
                blink_led(6, 0.3)
            
            if os.path.exists(h264_filename):
                try:
                    os.remove(h264_filename)
                except:
                    pass
        except:
            blink_led(6, 0.3)
    
    thread = threading.Thread(target=process, daemon=True)
    thread.start()

def main_program():
    """Função principal do programa"""
    global pose_detected_counter, last_predictions, recording, model_size, frame_counter, last_frame_time
    
    picam2 = None
    hailo = None
    circular_output = None
    
    try:
        hailo = Hailo(args.model)
        picam2 = Picamera2()
        
        main_size = (1280, 720)
        model_h, model_w, _ = hailo.get_input_shape()
        model_size = lores_size = (model_w, model_h)
        
        config = picam2.create_video_configuration(
            main={'size': main_size, 'format': 'XRGB8888'}, 
            lores={'size': lores_size, 'format': 'RGB888'}, 
            controls={'FrameRate': 30}
        )
        picam2.configure(config)
        

        ENABLE_PREVIEW = True
        if ENABLE_PREVIEW:
         from picamera2 import Preview
         picam2.post_callback = draw_predictions
         picam2.start_preview(Preview.QTGL)

        bitrate = 2000000
        encoder = H264Encoder(bitrate=bitrate)
        seconds_to_buffer = 60
        buffer_size_bytes = int(bitrate / 8 * seconds_to_buffer)
        circular_output = CircularOutput(buffersize=buffer_size_bytes)
        
        picam2.start_recording(encoder, circular_output)
        blink_led(3, 0.5)
        print("Sistema iniciado. Aguardando detecção da pose...")
        
        while True:
            try:
                # Captura frame com timeout
                frame = picam2.capture_array('lores')
                if frame is None:
                    continue
                
                # Debug: Print frame info once
                if frame_counter == 0:
                    print(f"Frame shape: {frame.shape}, Model size: {model_size}")
                
                
                # Processamento de detecção
                raw_detections = hailo.run(frame)
                if raw_detections is None:
                    continue
                
                # Debug: Print raw detection shapes
                frame_counter += 1
                if frame_counter % 30 == 0:  # Print every 30 frames
                    print(f"Frame {frame_counter}: Raw detections shapes:")
                    for key, value in raw_detections.items():
                        if isinstance(value, list):
                            print(f"  {key}: list with {len(value)} items")
                            for i, item in enumerate(value):
                                print(f"    [{i}]: {item.shape}")
                        else:
                            print(f"  {key}: {value.shape}")
                
                try:
                    last_predictions = postproc_yolov8_pose(1, raw_detections, model_size)
                except Exception as e:
                    if frame_counter % 30 == 0:
                        print(f"Postprocessing error: {e}")
                    last_predictions = None
                pose_found_this_frame = False
                
                # Debug: Print predictions every 30 frames
                if frame_counter % 30 == 0 and last_predictions:
                    print(f"Predictions keys: {list(last_predictions.keys())}")
                    if 'scores' in last_predictions and last_predictions['scores'][0] is not None:
                        scores = last_predictions['scores'][0]
                        num_detections = (scores > 0.1).sum()
                        print(f"  Found {num_detections} detections with confidence > 0.1")
                        if num_detections > 0:
                            max_score = scores.max()
                            print(f"  Highest confidence: {max_score:.3f}")
                
                if last_predictions and not recording:
                    try:
                        scores = last_predictions.get('scores', [None])[0]
                        keypoints = last_predictions.get('keypoints', [None])[0]
                        joint_scores = last_predictions.get('joint_scores', [None])[0]
                        
                        if scores is not None and keypoints is not None and joint_scores is not None:
                            for i in range(len(scores)):
                                if scores[i][0] > 0.5 and check_arms_crossed_above_head(keypoints[i], joint_scores[i].flatten()):
                                    pose_found_this_frame = True
                                    break
                    except:
                        pass
                
                # Controle de contador de pose
                if pose_found_this_frame:
                    pose_detected_counter += 1
                else:
                    pose_detected_counter = 0
                
                # Trigger de gravação
                if pose_detected_counter >= POSE_TRIGGER_FRAMES and not recording:
                    recording = True
                    print(f"✅ Pose confirmada! Gravando...")
                    
                    led.on()
                    time.sleep(2)
                    led.off()
                    
                    base_filename = datetime.now().strftime("gravacao_%Y-%m-%d_%H-%M-%S")
                    h264_filename = base_filename + ".h264"
                    mp4_filename = base_filename + ".mp4"
                    
                    try:
                        circular_output.fileoutput = h264_filename
                        circular_output.start()
                        circular_output.stop()
                        recording = False
                        # Processa vídeo em thread separada
                        process_video_async(h264_filename, mp4_filename)
                        
                    except Exception as e:
                        blink_led(6, 0.3)
                    
                    pose_detected_counter = 0
                    recording = False
                
            except Exception as e:
                # Erro no loop principal - continua executando
                time.sleep(0.1)  # Pequena pausa para evitar loop intenso
                continue
                
    except KeyboardInterrupt:
        print("\nPrograma interrompido pelo usuário.")
        return
    except Exception as e:
        print(f"❌ Erro crítico: {e}")
        raise
    finally:
        # Cleanup robusto
        try:
            if picam2 and picam2.is_open:
                picam2.stop_recording()
                picam2.close()
        except:
            pass
        try:
            if hailo:
                hailo.close()
        except:
            pass
        led.off()

if __name__ == "__main__":
    while True:  # Loop principal para reiniciar em caso de erro
        try:
            main_program()
            break  # Sai se terminou normalmente
        except KeyboardInterrupt:
            print("\nPrograma interrompido pelo usuário.")
            led.off()
            break
        except Exception as e:
            print(f"❌ Erro crítico detectado. Reiniciando sistema em 5 segundos...")
            led.off()
            time.sleep(5)
            continue
