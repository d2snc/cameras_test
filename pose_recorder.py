#!/usr/bin/env python3
import argparse
import cv2
import time
from datetime import datetime
import subprocess
import os
import logging
import numpy as np
from gpiozero import LED
from picamera2 import MappedArray, Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput

# Configurar logging mais detalhado
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pose_detection.log')
    ]
)
logger = logging.getLogger(__name__)

led = LED(17)

def led_startup_signal():
    """LED pisca 3 vezes por meio segundo - sistema iniciado"""
    for _ in range(3):
        led.on()
        time.sleep(0.5)
        led.off()
        time.sleep(0.2)

def led_recording_signal():
    """LED pisca 1 vez por 1 segundo - vai gravar"""
    led.on()
    time.sleep(1)
    led.off()

def led_success_signal():
    """LED pisca 2 vezes por meio segundo - gravação bem-sucedida"""
    for _ in range(2):
        led.on()
        time.sleep(0.5)
        led.off()
        time.sleep(0.2)

def led_error_signal():
    """LED pisca 5 vezes por 0.3 segundos - gravação falhou"""
    for _ in range(5):
        led.on()
        time.sleep(0.3)
        led.off()
        time.sleep(0.1)

def led_critical_error_signal():
    """LED pisca 10 vezes por 0.3 segundos - erro crítico"""
    for _ in range(10):
        led.on()
        time.sleep(0.3)
        led.off()
        time.sleep(0.1)

parser = argparse.ArgumentParser(description='Sistema de Gravação com Detecção de Movimento')
args = parser.parse_args()

# Variáveis para detecção de movimento
MOTION_TRIGGER_FRAMES = 30  # Frames consecutivos com movimento para trigger
motion_detected_counter = 0
previous_frame = None
recording = False

def safe_remove_file(filepath):
    """Remove arquivo de forma segura"""
    try:
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            os.remove(filepath)
            logger.info(f"Arquivo removido: {filepath} (era {file_size} bytes)")
            return True
        else:
            logger.warning(f"Tentativa de remover arquivo inexistente: {filepath}")
    except Exception as e:
        logger.error(f"Erro ao remover arquivo {filepath}: {e}")
    return False

def check_ffmpeg_available():
    """Verifica se FFmpeg está disponível"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.debug(f"FFmpeg version check: return code {result.returncode}")
        
        result2 = subprocess.run(['ffprobe', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.debug(f"FFprobe version check: return code {result2.returncode}")
        
        if result.returncode == 0 and result2.returncode == 0:
            logger.info("FFmpeg e FFprobe disponíveis")
            return True
        else:
            logger.error("FFmpeg ou FFprobe retornaram erro")
            return False
    except FileNotFoundError as e:
        logger.error(f"FFmpeg ou FFprobe não encontrado: {e}")
        return False
    except Exception as e:
        logger.error(f"Erro ao verificar FFmpeg: {e}")
        return False

def get_video_duration(filepath):
    """Obtém duração do vídeo de forma robusta"""
    logger.debug(f"Tentando obter duração de: {filepath}")
    
    if not os.path.exists(filepath):
        logger.error(f"Arquivo não existe: {filepath}")
        return None
    
    file_size = os.path.getsize(filepath)
    logger.debug(f"Tamanho do arquivo: {file_size} bytes")
    
    if file_size == 0:
        logger.error(f"Arquivo está vazio: {filepath}")
        return None
    
    try:
        # Método 1: ffprobe
        command = [
            'ffprobe', '-v', 'quiet',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            filepath
        ]
        logger.debug(f"Executando comando: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        logger.debug(f"Return code: {result.returncode}, stdout: '{result.stdout}', stderr: '{result.stderr}'")
        
        if result.returncode == 0 and result.stdout.strip():
            duration = float(result.stdout.strip())
            logger.info(f"Duração detectada: {duration:.1f} segundos")
            return duration
    except Exception as e:
        logger.warning(f"Método 1 falhou: {e}")
    
    try:
        # Método 2: ffprobe alternativo
        command = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=duration',
            '-of', 'csv=p=0',
            filepath
        ]
        logger.debug(f"Executando comando alternativo: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        logger.debug(f"Return code: {result.returncode}, stdout: '{result.stdout}', stderr: '{result.stderr}'")
        
        if result.returncode == 0 and result.stdout.strip():
            duration = float(result.stdout.strip())
            logger.info(f"Duração detectada (método 2): {duration:.1f} segundos")
            return duration
    except Exception as e:
        logger.warning(f"Método 2 falhou: {e}")
    
    logger.error("Não foi possível determinar a duração do vídeo")
    return None

def save_buffer_to_file(circular_output, filepath):
    """Salva buffer circular para arquivo de forma robusta"""
    logger.debug(f"Iniciando salvamento do buffer para: {filepath}")
    
    try:
        # Verificar estado do circular_output
        buffer_size = circular_output.tell()
        logger.debug(f"Tamanho atual do buffer: {buffer_size} bytes")
        
        if buffer_size == 0:
            logger.error("Buffer circular está vazio!")
            return False
        
        with open(filepath, "wb") as f:
            logger.debug("Arquivo aberto para escrita")
            circular_output.fileoutput = f
            logger.debug("fileoutput configurado")
            circular_output.copy_to_file()
            logger.debug("copy_to_file executado")
        
        # Verificar se arquivo foi criado e tem tamanho > 0
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            if file_size > 0:
                size_mb = file_size / (1024 * 1024)
                logger.info(f"Buffer salvo com sucesso: {filepath} ({size_mb:.1f} MB)")
                return True
            else:
                logger.error(f"Arquivo foi criado mas está vazio: {filepath}")
                return False
        else:
            logger.error(f"Arquivo não foi criado: {filepath}")
            return False
            
    except Exception as e:
        logger.error(f"Erro ao salvar buffer: {e}")
        logger.exception("Stack trace completo:")
        return False

def trim_video_to_duration(input_file, output_file, target_duration, fps=30):
    """Corta vídeo para duração específica com diagnóstico detalhado"""
    
    logger.debug(f"Iniciando corte: {input_file} -> {output_file}, duração: {target_duration}s")
    
    # Verificar se arquivo de entrada existe
    if not os.path.exists(input_file):
        logger.error(f"Arquivo de entrada não encontrado: {input_file}")
        return False
    
    input_size = os.path.getsize(input_file)
    logger.debug(f"Tamanho do arquivo de entrada: {input_size} bytes")
    
    duration = get_video_duration(input_file)
    
    if duration is None:
        logger.warning("Usando corte simples sem detecção de duração")
        commands_to_try = [
            ['ffmpeg', '-y', '-i', input_file, '-t', str(target_duration), '-c:v', 'copy', output_file],
            ['ffmpeg', '-y', '-i', input_file, '-t', str(target_duration), '-c:v', 'libx264', '-preset', 'ultrafast', output_file]
        ]
    else:
        if duration > target_duration:
            start_time = duration - target_duration
            logger.info(f"Cortando últimos {target_duration} segundos (início em {start_time:.1f}s)")
            commands_to_try = [
                ['ffmpeg', '-y', '-ss', str(start_time), '-i', input_file, '-t', str(target_duration), '-c:v', 'copy', output_file],
                ['ffmpeg', '-y', '-ss', str(start_time), '-i', input_file, '-t', str(target_duration), '-c:v', 'libx264', '-preset', 'ultrafast', output_file]
            ]
        else:
            logger.info(f"Vídeo tem {duration:.1f}s, menor que {target_duration}s - usando tudo")
            commands_to_try = [
                ['ffmpeg', '-y', '-i', input_file, '-c:v', 'copy', output_file],
                ['ffmpeg', '-y', '-i', input_file, '-c:v', 'libx264', '-preset', 'ultrafast', output_file]
            ]
    
    # Tentar cada comando
    for i, command in enumerate(commands_to_try, 1):
        try:
            logger.info(f"Tentativa {i}: {' '.join(command)}")
            
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=180,  # 3 minutos timeout
                check=False  # Não levantar exceção automaticamente
            )
            
            logger.debug(f"Return code: {result.returncode}")
            logger.debug(f"Stdout: {result.stdout}")
            logger.debug(f"Stderr: {result.stderr}")
            
            if result.returncode == 0:
                # Verificar se arquivo foi criado com sucesso
                if os.path.exists(output_file):
                    output_size = os.path.getsize(output_file)
                    if output_size > 0:
                        size_mb = output_size / (1024 * 1024)
                        logger.info(f"✅ Vídeo cortado com sucesso: {output_file} ({size_mb:.1f} MB)")
                        return True
                    else:
                        logger.warning(f"Arquivo criado mas vazio: {output_file}")
                else:
                    logger.warning(f"Comando executou mas arquivo não foi criado: {output_file}")
            else:
                logger.error(f"FFmpeg retornou código {result.returncode}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout na tentativa {i}")
            continue
        except Exception as e:
            logger.error(f"Erro inesperado na tentativa {i}: {e}")
            continue
    
    logger.error("Todas as tentativas de corte falharam")
    return False

def convert_to_mp4(input_file, output_file, fps=30):
    """Converte H264 para MP4 com diagnóstico detalhado"""
    
    logger.debug(f"Iniciando conversão: {input_file} -> {output_file}")
    
    if not os.path.exists(input_file):
        logger.error(f"Arquivo de entrada não encontrado: {input_file}")
        return False
    
    input_size = os.path.getsize(input_file)
    logger.debug(f"Tamanho do arquivo de entrada: {input_size} bytes")
    
    commands_to_try = [
        ['ffmpeg', '-y', '-framerate', str(fps), '-i', input_file, '-c:v', 'copy', output_file],
        ['ffmpeg', '-y', '-i', input_file, '-c:v', 'copy', output_file],
        ['ffmpeg', '-y', '-i', input_file, '-c:v', 'libx264', '-preset', 'ultrafast', output_file]
    ]
    
    for i, command in enumerate(commands_to_try, 1):
        try:
            logger.info(f"Conversão - Tentativa {i}: {' '.join(command)}")
            
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=240,  # 4 minutos
                check=False
            )
            
            logger.debug(f"Return code: {result.returncode}")
            logger.debug(f"Stdout: {result.stdout}")
            logger.debug(f"Stderr: {result.stderr}")
            
            if result.returncode == 0:
                if os.path.exists(output_file):
                    output_size = os.path.getsize(output_file)
                    if output_size > 0:
                        size_mb = output_size / (1024 * 1024)
                        logger.info(f"✅ Conversão bem-sucedida: {output_file} ({size_mb:.1f} MB)")
                        return True
                    else:
                        logger.warning(f"Arquivo MP4 criado mas vazio: {output_file}")
                else:
                    logger.warning(f"Comando executou mas arquivo MP4 não foi criado")
            else:
                logger.error(f"FFmpeg conversão retornou código {result.returncode}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout na conversão tentativa {i}")
            continue
        except Exception as e:
            logger.error(f"Erro inesperado na conversão tentativa {i}: {e}")
            continue
    
    logger.error("Todas as tentativas de conversão falharam")
    return False

def detect_motion(frame, threshold=1000):
    """Detecta movimento simples baseado na diferença entre frames"""
    global previous_frame
    
    # Converter para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if previous_frame is None:
        previous_frame = gray
        return False
    
    # Calcular diferença absoluta
    frame_delta = cv2.absdiff(previous_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Contar pixels brancos (movimento)
    motion_pixels = cv2.countNonZero(thresh)
    
    # Atualizar frame anterior
    previous_frame = gray
    
    return motion_pixels > threshold

# Verificar FFmpeg antes de iniciar
logger.info("Verificando FFmpeg...")
if not check_ffmpeg_available():
    logger.error("FFmpeg não disponível - encerrando")
    led_critical_error_signal()
    exit(1)

picam2 = Picamera2()

try:
    logger.info("Inicializando sistema...")
    
    main_size = (1280, 720)
    config = picam2.create_video_configuration(
        main={'size': main_size, 'format': 'XRGB8888'}, 
        lores={'size': (320, 240), 'format': 'RGB888'},  # Para detecção de movimento
        controls={'FrameRate': 30}
    )
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
    
    logger.info(f"Gravação iniciada - Buffer: {buffer_size_bytes / (1024*1024):.0f} MB, Bitrate: {bitrate}")
    
    # LED indica sistema iniciado com sucesso
    logger.info("✅ Sistema iniciado com sucesso")
    led_startup_signal()
    
    logger.info("🚀 Sistema pronto. Aguardando detecção de movimento...")
    
    while True:
        # Capturar frame em baixa resolução para detecção de movimento
        frame = picam2.capture_array('lores')
        
        # Detectar movimento
        motion_detected = detect_motion(frame)
        
        if motion_detected and not recording:
            motion_detected_counter += 1
            if motion_detected_counter % 10 == 0:  # Log a cada 10 frames
                logger.debug(f"Movimento detectado por {motion_detected_counter} frames")
        else:
            motion_detected_counter = 0
            
        if motion_detected_counter >= MOTION_TRIGGER_FRAMES and not recording:
            recording = True
            logger.info(f"✅ Movimento confirmado por {MOTION_TRIGGER_FRAMES} frames! Iniciando gravação...")
            
            # Verificar estado do buffer antes de gravar
            buffer_current = circular_output.tell()
            logger.info(f"Buffer atual: {buffer_current / (1024*1024):.1f} MB")
            
            # LED indica que vai gravar
            led_recording_signal()
            
            base_filename = datetime.now().strftime("gravacao_%Y-%m-%d_%H-%M-%S")
            h264_temp_filename = base_filename + "_buffer.h264"
            h264_trimmed_filename = base_filename + "_trimmed.h264"
            mp4_filename = base_filename + ".mp4"
            
            success = False
            temp_files = []
            step_failed = ""
            
            try:
                # PASSO 1: Salvar buffer circular
                logger.info("=== PASSO 1: Salvando buffer circular ===")
                if save_buffer_to_file(circular_output, h264_temp_filename):
                    temp_files.append(h264_temp_filename)
                    step_failed = "corte"
                    
                    # PASSO 2: Cortar para duração exata
                    logger.info("=== PASSO 2: Cortando vídeo ===")
                    if trim_video_to_duration(h264_temp_filename, h264_trimmed_filename, target_duration_seconds, fps):
                        temp_files.append(h264_trimmed_filename)
                        step_failed = "conversão"
                        
                        # PASSO 3: Converter para MP4
                        logger.info("=== PASSO 3: Convertendo para MP4 ===")
                        if convert_to_mp4(h264_trimmed_filename, mp4_filename, fps):
                            success = True
                            logger.info(f"✅ Gravação finalizada: {mp4_filename}")
                        else:
                            logger.error("Falha na conversão para MP4")
                    else:
                        logger.error("Falha no corte do vídeo")
                else:
                    step_failed = "salvamento do buffer"
                    logger.error("Falha ao salvar buffer")
                    
                if success:
                    # LED indica gravação bem-sucedida
                    led_success_signal()
                    logger.info("✅ Gravação concluída com sucesso")
                else:
                    # LED indica falha na gravação
                    led_error_signal()
                    logger.error(f"❌ Processo de gravação falhou na etapa: {step_failed}")
                    
            except Exception as e:
                logger.error(f"Erro inesperado durante processamento: {e}")
                logger.exception("Stack trace completo:")
                led_error_signal()
                
            finally:
                # Limpar arquivos temporários
                logger.info("=== Limpando arquivos temporários ===")
                for temp_file in temp_files:
                    safe_remove_file(temp_file)
                
                logger.info("-> Sistema pronto para nova detecção.")
                motion_detected_counter = 0
                recording = False

except KeyboardInterrupt:
    logger.info("Programa interrompido pelo usuário")
except Exception as e:
    logger.error(f"Erro crítico: {e}")
    logger.exception("Stack trace completo:")
    # LED indica erro crítico
    led_critical_error_signal()
finally:
    try:
        if picam2.is_open:
            picam2.stop_recording()
        logger.info("Programa encerrado.")
    except:
        pass
