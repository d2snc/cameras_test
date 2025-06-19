#!/usr/bin/env python3
import argparse
import cv2
import time
from datetime import datetime
import subprocess
import os
import logging
from gpiozero import LED
from pose_utils import postproc_yolov8_pose
from picamera2 import MappedArray, Picamera2 #, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput
from picamera2.devices import Hailo

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def safe_remove_file(filepath):
    """Remove arquivo de forma segura"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Arquivo removido: {filepath}")
            return True
    except Exception as e:
        logger.error(f"Erro ao remover arquivo {filepath}: {e}")
    return False

def check_ffmpeg_available():
    """Verifica se FFmpeg está disponível"""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        subprocess.run(['ffprobe', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FFmpeg ou FFprobe não encontrado. Instale com: sudo apt install ffmpeg")
        return False

def get_video_duration(filepath):
    """Obtém duração do vídeo de forma robusta"""
    try:
        # Método 1: ffprobe
        command = [
            'ffprobe', '-v', 'quiet', 
            '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1', 
            filepath
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            duration = float(result.stdout.strip())
            logger.info(f"Duração detectada: {duration:.1f} segundos")
            return duration
            
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
        logger.warning(f"Método 1 falhou: {e}")
    
    try:
        # Método 2: ffprobe alternativo
        command = [
            'ffprobe', '-v', 'quiet', 
            '-select_streams', 'v:0', 
            '-show_entries', 'stream=duration', 
            '-of', 'csv=p=0', 
            filepath
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            duration = float(result.stdout.strip())
            logger.info(f"Duração detectada (método 2): {duration:.1f} segundos")
            return duration
            
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
        logger.warning(f"Método 2 falhou: {e}")
    
    logger.error("Não foi possível determinar a duração do vídeo")
    return None

def save_buffer_to_file(circular_output, filepath):
    """Salva buffer circular para arquivo de forma robusta"""
    try:
        with open(filepath, "wb") as f:
            circular_output.fileoutput = f
            circular_output.copy_to_file()
        
        # Verificar se arquivo foi criado e tem tamanho > 0
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            logger.info(f"Buffer salvo: {filepath} ({size_mb:.1f} MB)")
            return True
        else:
            logger.error(f"Arquivo não foi criado ou está vazio: {filepath}")
            return False
            
    except Exception as e:
        logger.error(f"Erro ao salvar buffer: {e}")
        return False

def trim_video_to_duration(input_file, output_file, target_duration, fps=30):
    """Corta vídeo para duração específica com múltiplas tentativas"""
    
    # Verificar se arquivo de entrada existe
    if not os.path.exists(input_file):
        logger.error(f"Arquivo de entrada não encontrado: {input_file}")
        return False
    
    duration = get_video_duration(input_file)
    
    if duration is None:
        logger.warning("Usando corte simples sem detecção de duração")
        commands_to_try = [
            # Método 1: Corte simples
            ['ffmpeg', '-y', '-i', input_file, '-t', str(target_duration), '-c:v', 'copy', output_file],
            # Método 2: Com framerate
            ['ffmpeg', '-y', '-framerate', str(fps), '-i', input_file, '-t', str(target_duration), '-c:v', 'copy', output_file],
            # Método 3: Recodificação se cópia falhar
            ['ffmpeg', '-y', '-i', input_file, '-t', str(target_duration), '-c:v', 'libx264', '-preset', 'fast', output_file]
        ]
    else:
        if duration > target_duration:
            # Pegar os últimos N segundos
            start_time = duration - target_duration
            logger.info(f"Cortando últimos {target_duration} segundos (início em {start_time:.1f}s)")
            commands_to_try = [
                # Método 1: Corte dos últimos segundos
                ['ffmpeg', '-y', '-ss', str(start_time), '-i', input_file, '-t', str(target_duration), '-c:v', 'copy', output_file],
                # Método 2: Sem copy (recodificação)
                ['ffmpeg', '-y', '-ss', str(start_time), '-i', input_file, '-t', str(target_duration), '-c:v', 'libx264', '-preset', 'fast', output_file]
            ]
        else:
            logger.info(f"Vídeo tem {duration:.1f}s, menor que {target_duration}s - usando tudo")
            commands_to_try = [
                # Método 1: Cópia direta
                ['ffmpeg', '-y', '-i', input_file, '-c:v', 'copy', output_file],
                # Método 2: Com recodificação
                ['ffmpeg', '-y', '-i', input_file, '-c:v', 'libx264', '-preset', 'fast', output_file]
            ]
    
    # Tentar cada comando
    for i, command in enumerate(commands_to_try, 1):
        try:
            logger.info(f"Tentativa {i}: {' '.join(command)}")
            result = subprocess.run(
                command, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.PIPE, 
                text=True, 
                timeout=120,  # 2 minutos timeout
                check=True
            )
            
            # Verificar se arquivo foi criado com sucesso
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                size_mb = os.path.getsize(output_file) / (1024 * 1024)
                logger.info(f"✅ Vídeo cortado com sucesso: {output_file} ({size_mb:.1f} MB)")
                return True
            else:
                logger.warning(f"Comando executou mas arquivo não foi criado: tentativa {i}")
                continue
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout na tentativa {i}")
            continue
        except subprocess.CalledProcessError as e:
            logger.error(f"Tentativa {i} falhou: {e}")
            if e.stderr:
                logger.error(f"Stderr: {e.stderr}")
            continue
        except Exception as e:
            logger.error(f"Erro inesperado na tentativa {i}: {e}")
            continue
    
    logger.error("Todas as tentativas de corte falharam")
    return False

def convert_to_mp4(input_file, output_file, fps=30):
    """Converte H264 para MP4 com múltiplas tentativas"""
    
    if not os.path.exists(input_file):
        logger.error(f"Arquivo de entrada não encontrado: {input_file}")
        return False
    
    commands_to_try = [
        # Método 1: Cópia simples
        ['ffmpeg', '-y', '-framerate', str(fps), '-i', input_file, '-c:v', 'copy', output_file],
        # Método 2: Sem framerate especificado
        ['ffmpeg', '-y', '-i', input_file, '-c:v', 'copy', output_file],
        # Método 3: Com recodificação
        ['ffmpeg', '-y', '-i', input_file, '-c:v', 'libx264', '-preset', 'fast', output_file],
        # Método 4: Formato mais compatível
        ['ffmpeg', '-y', '-i', input_file, '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p', output_file]
    ]
    
    for i, command in enumerate(commands_to_try, 1):
        try:
            logger.info(f"Conversão - Tentativa {i}")
            result = subprocess.run(
                command, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.PIPE, 
                text=True, 
                timeout=180,  # 3 minutos
                check=True
            )
            
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                size_mb = os.path.getsize(output_file) / (1024 * 1024)
                logger.info(f"✅ Conversão bem-sucedida: {output_file} ({size_mb:.1f} MB)")
                return True
                
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"Tentativa {i} de conversão falhou: {e}")
            continue
        except Exception as e:
            logger.error(f"Erro inesperado na conversão tentativa {i}: {e}")
            continue
    
    logger.error("Todas as tentativas de conversão falharam")
    return False

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

# Verificar FFmpeg antes de iniciar
if not check_ffmpeg_available():
    exit(1)

picam2 = Picamera2()

try:
    with Hailo(args.model) as hailo:
        main_size = (1280, 720)
        model_h, model_w, _ = hailo.get_input_shape()
        model_size = lores_size = (model_w, model_h)
        config = picam2.create_video_configuration(main={'size': main_size, 'format': 'XRGB8888'}, lores={'size': lores_size, 'format': 'RGB888'}, controls={'FrameRate': 30})
        picam2.configure(config)
        
        # Configuração do buffer
        fps = 30
        target_duration_seconds = 60
        bitrate = 2000000
        encoder = H264Encoder(bitrate=bitrate)
        
        # Buffer generoso para garantir mais de 60 segundos
        buffer_size_bytes = 150 * 1024 * 1024  # 150 MB
        circular_output = CircularOutput(buffersize=buffer_size_bytes)
        picam2.start_recording(encoder, circular_output)
        
        # pisca o led 3 vezes para indicar que o sistema está pronto
        for _ in range(3):
            led.on()
            time.sleep(1)
            led.off()
            time.speep(1)
        logger.info("🚀 Sistema iniciado. Aguardando detecção da pose...")
        
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
                led.on()
                time.sleep(0.1)
                led.off()
            else:
                pose_detected_counter = 0
                
            if pose_detected_counter >= POSE_TRIGGER_FRAMES and not recording:
                recording = True
                logger.info(f"✅ Pose confirmada por {POSE_TRIGGER_FRAMES} frames! Acionando LED e gravação...")
                led.on()
                time.sleep(0.5)
                led.off()
                time.speep(0.5)
                led.on()
                time.sleep(0.5)
                led.off()
           
                
                base_filename = datetime.now().strftime("gravacao_%Y-%m-%d_%H-%M-%S")
                h264_temp_filename = base_filename + "_buffer.h264"
                h264_trimmed_filename = base_filename + "_trimmed.h264"
                mp4_filename = base_filename + ".mp4"
                
                success = False
                temp_files = []
                
                try:
                    # PASSO 1: Salvar buffer circular
                    logger.info("Salvando buffer circular...")
                    if save_buffer_to_file(circular_output, h264_temp_filename):
                        temp_files.append(h264_temp_filename)
                        
                        # PASSO 2: Cortar para duração exata
                        logger.info(f"Cortando para {target_duration_seconds} segundos...")
                        if trim_video_to_duration(h264_temp_filename, h264_trimmed_filename, target_duration_seconds, fps):
                            temp_files.append(h264_trimmed_filename)
                            
                            # PASSO 3: Converter para MP4
                            logger.info("Convertendo para MP4...")
                            if convert_to_mp4(h264_trimmed_filename, mp4_filename, fps):
                                success = True
                                logger.info(f"✅ Gravação finalizada: {mp4_filename}")
                            else:
                                logger.error("Falha na conversão para MP4")
                        else:
                            logger.error("Falha no corte do vídeo")
                    else:
                        logger.error("Falha ao salvar buffer")
                        
                    if not success:
                        logger.error("❌ Processo de gravação falhou completamente")
                        
                except Exception as e:
                    logger.error(f"Erro inesperado durante processamento: {e}")
                    
                finally:
                    # Limpar arquivos temporários
                    for temp_file in temp_files:
                        safe_remove_file(temp_file)
                    
                    logger.info("-> Sistema pronto para nova detecção.")
                    pose_detected_counter = 0
                    recording = False
                    led.on()
                    time.sleep(3)
                    led.off()

except Exception as e:
    logger.error(f"Erro crítico: {e}")
    led.on()
    time.sleep(0.3)
    led.off()
    time.speep(0.3)
    led.on()
    time.sleep(0.3)
    led.off()
    time.sleep(0.3)
    led.off()
    time.speep(0.3)
    led.on()
    time.sleep(0.3)
    led.off()
    time.sleep(0.3)
    led.off()
    time.speep(0.3)
    led.on()
    time.sleep(0.3)
    led.off()
finally:
    if picam2.is_open:
        picam2.stop_recording()
    logger.info("Programa encerrado.")
