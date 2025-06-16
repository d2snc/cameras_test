import cv2
import numpy as np
import time
from picamera2 import Picamera2
from collections import deque
import threading
import datetime
import os

# --- Importações da Hailo ---
from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams,
                            InputVStream, OutputVStream)

# --- Configurações ---
HEF_PATH = 'yolov8s_pose.hef'
VIDEO_DURATION_SECONDS = 20
CAMERA_RESOLUTION = (640, 480) # Resolução para a câmera e modelo
COOLDOWN_SECONDS = 30 # Tempo de espera em segundos após salvar um vídeo

# --- Mapeamento de Pontos-Chave (Keypoints) do YOLOv8-Pose ---
# (Verifique se corresponde ao seu modelo, este é o padrão COCO)
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_WRIST = 9
RIGHT_WRIST = 10

def setup_hailo_device(hef_path):
    """Inicializa o VDevice da Hailo e carrega o HEF."""
    print("🔎 Inicializando dispositivo Hailo e carregando HEF...")
    devices = VDevice.scan()
    if not devices:
        raise RuntimeError("Nenhum dispositivo Hailo encontrado.")
    
    target = VDevice(devices[0])
    
    # Carrega o modelo compilado (HEF)
    hef = HEF(hef_path)
    
    # Configura os streams de entrada/saída
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    
    # Obtenha informações sobre input e output
    input_vstreams = InputVStream.make(network_group, transfer_mode=HailoStreamInterface.PCIe)
    output_vstreams = OutputVStream.make(network_group, transfer_mode=HailoStreamInterface.PCIe)
    
    print("✅ Dispositivo Hailo pronto.")
    return target, hef, input_vstreams[0], output_vstreams[0]


def save_video_from_buffer(video_buffer, resolution, fps):
    """Salva os frames do buffer em um arquivo de vídeo."""
    if not video_buffer:
        print("⚠️ Buffer de vídeo vazio, nada para salvar.")
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"gravacao_{timestamp}.mp4"
    
    print(f"🎬 Salvando vídeo: {filename}")
    
    # Usa o codec 'mp4v' para compatibilidade
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, resolution)

    for frame in list(video_buffer):
        out.write(frame)

    out.release()
    print(f"✅ Vídeo salvo com sucesso!")


def check_arms_crossed_above_head(keypoints, frame_shape):
    """
    Verifica se a pose 'braços cruzados acima da cabeça' é detectada.
    `keypoints` é um array com formato (17, 3) para [x, y, confidence].
    """
    if keypoints.shape[0] < 17:
        return False

    h, w = frame_shape

    # Extrai os pontos-chave necessários
    nose_pt = keypoints[NOSE]
    left_shoulder_pt = keypoints[LEFT_SHOULDER]
    right_shoulder_pt = keypoints[RIGHT_SHOULDER]
    left_wrist_pt = keypoints[LEFT_WRIST]
    right_wrist_pt = keypoints[RIGHT_WRIST]

    # Verifica a confiança (só analisa se os pontos-chave foram detectados com certeza)
    min_confidence = 0.5
    if any(pt[2] < min_confidence for pt in [nose_pt, left_shoulder_pt, right_shoulder_pt, left_wrist_pt, right_wrist_pt]):
        return False

    # Condição 1: Pulsos acima dos ombros e do nariz (coordenada Y menor)
    wrist_above_shoulders = left_wrist_pt[1] < left_shoulder_pt[1] and right_wrist_pt[1] < right_shoulder_pt[1]
    wrist_above_nose = left_wrist_pt[1] < nose_pt[1] and right_wrist_pt[1] < nose_pt[1]

    # Condição 2: Pulsos cruzados (pulso esquerdo mais à direita que o direito)
    wrists_crossed = left_wrist_pt[0] > right_wrist_pt[0]

    return wrist_above_shoulders and wrist_above_nose and wrists_crossed


def main():
    # --- Inicialização ---
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": CAMERA_RESOLUTION, "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    print("📷 Câmera iniciada.")

    target, hef, input_vstream, output_vstream = setup_hailo_device(HEF_PATH)
    
    # Determina o FPS da câmera e o tamanho do buffer
    # A captura e o processamento podem não atingir o FPS máximo, então medimos.
    # Por segurança, calculamos o buffer com um FPS estimado de 15.
    estimated_fps = 15
    buffer_size = VIDEO_DURATION_SECONDS * estimated_fps
    video_buffer = deque(maxlen=buffer_size)

    last_trigger_time = 0

    print("🚀 Iniciando loop de detecção...")
    try:
        while True:
            # Captura e pré-processamento do frame
            frame = picam2.capture_array()
            
            # Adiciona o frame ao buffer circular
            video_buffer.append(frame)

            # Envia o frame para inferência no Hailo-8L
            with input_vstream.write_async(frame):
                # Enquanto a inferência acontece, podemos fazer outras coisas se necessário
                # Aqui, vamos esperar pelo resultado
                result_raw = output_vstream.read(timeout=1000) # Timeout de 1s
            
            # O pós-processamento depende da saída exata do seu modelo YOLOv8.
            # Geralmente é uma combinação de caixas delimitadoras e pontos-chave.
            # Assumindo que a saída já foi processada para extrair os keypoints
            # de uma pessoa. Esta parte pode precisar de ajuste!
            
            # **ADAPTE ESTA PARTE CONFORME A SAÍDA DO SEU MODELO**
            # Exemplo: Vamos assumir que `result_raw` contém um array de detecções,
            # e cada detecção tem a pose. Para simplificar, pegamos a primeira.
            # A forma da saída pode ser (1, 84, 8400) ou similar.
            # Você precisará de uma função de pós-processamento para decodificar isso.
            
            # Placeholder para a lógica de pós-processamento real
            # Aqui, simulamos que `post_process` retorna uma lista de poses detectadas
            # e cada pose é um array (17, 3) de [x, y, conf].
            #detections = post_process_yolov8_pose(result_raw)
            
            # --- SIMULAÇÃO DE DETECÇÃO PARA TESTE ---
            # Para testar sem a lógica de pós-processamento complexa, você pode
            # pular a inferência e desenhar pontos-chave falsos na tela.
            # Esta seção deve ser substituída pela sua lógica de pós-processamento real.
            
            # --- LÓGICA DE DETECÇÃO REAL ---
            # (Aqui você implementaria a decodificação da saída `result_raw`)
            # Por enquanto, vamos assumir que não há detecção para o loop continuar.
            detections = [] 

            # Itera sobre todas as pessoas detectadas no frame
            for person_keypoints in detections:
                if time.time() - last_trigger_time < COOLDOWN_SECONDS:
                    continue # Respeita o tempo de espera

                is_triggered = check_arms_crossed_above_head(person_keypoints, frame.shape[:2])

                if is_triggered:
                    print(f"🚨 Gesto detectado! Salvando {VIDEO_DURATION_SECONDS} segundos de vídeo.")
                    last_trigger_time = time.time()
                    
                    # Salva o vídeo em uma thread separada para não bloquear a captura
                    save_thread = threading.Thread(target=save_video_from_buffer, args=(list(video_buffer), CAMERA_RESOLUTION, estimated_fps))
                    save_thread.start()
                    break # Sai do loop de detecções para este frame
            
            # Opcional: Mostrar o vídeo na tela para depuração
            # cv2.imshow("Camera Feed", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    except KeyboardInterrupt:
        print("\n🛑 Parando o script.")
    finally:
        picam2.stop()
        # cv2.destroyAllWindows()
        print("Recursos liberados.")


if __name__ == "__main__":
    if not os.path.exists(HEF_PATH):
        print(f"ERRO: Arquivo {HEF_PATH} não encontrado. Certifique-se de que ele está no mesmo diretório do script.")
    else:
        main()