#!/usr/bin/env python3

import argparse
import cv2
import time
from datetime import datetime

# Importa as funções de pós-processamento para o YOLOv8 Pose
from pose_utils import postproc_yolov8_pose

# Importa as bibliotecas da Picamera2 e do Hailo
from picamera2 import MappedArray, Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput
from picamera2.devices import Hailo

# --- Configuração Inicial ---

# Analisador de argumentos para especificar o caminho do modelo
parser = argparse.ArgumentParser(description='Detecção de Pose com Gravação por Gatilho no Hailo')
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
last_predictions = None  # Armazena a última predição da IA
recording = False        # Flag para controlar o estado da gravação

# --- Funções Principais ---

def check_arms_crossed_above_head(keypoints, joint_scores, threshold=0.5):
    """
    Verifica se a pose 'braços cruzados acima da cabeça' foi detectada.
    A lógica verifica se os pulsos estão posicionados acima dos ombros e do nariz.
    """
    # Garante que os pontos-chave relevantes tenham uma pontuação de confiança mínima
    if all(joint_scores[i] > threshold for i in [L_WRIST, R_WRIST, L_SHOULDER, R_SHOULDER, NOSE]):
        left_wrist_y = keypoints[L_WRIST][1]
        right_wrist_y = keypoints[R_WRIST][1]
        left_shoulder_y = keypoints[L_SHOULDER][1]
        right_shoulder_y = keypoints[R_SHOULDER][1]
        nose_y = keypoints[NOSE][1]

        # Condição: ambos os pulsos devem estar acima da média da altura dos ombros e também acima do nariz
        if left_wrist_y < nose_y and right_wrist_y < nose_y and \
           left_wrist_y < (left_shoulder_y + right_shoulder_y) / 2 and \
           right_wrist_y < (left_shoulder_y + right_shoulder_y) / 2:
            return True
    return False

def visualize_pose_estimation_result(results, image, model_size, detection_threshold=0.5, joint_threshold=0.5):
    """
    Desenha os resultados da detecção de pose (caixas delimitadoras e esqueleto) na imagem.
    """
    image_size = (image.shape[1], image.shape[0])

    def scale_coord(coord):
        return tuple([int(c * t / f) for c, f, t in zip(coord, model_size, image_size)])

    if not results or 'bboxes' not in results:
        return

    # Extrai os dados da predição (assumindo batch size = 1)
    bboxes, scores, keypoints, joint_scores = (
        results['bboxes'][0], results['scores'][0], results['keypoints'][0], results['joint_scores'][0])

    for detection_box, detection_score, detection_keypoints, detection_keypoints_score in \
            zip(bboxes, scores, keypoints, joint_scores):
        if detection_score[0] < detection_threshold:
            continue

        # Desenha a caixa delimitadora
        coord_min = scale_coord(detection_box[:2])
        coord_max = scale_coord(detection_box[2:])
        cv2.rectangle(image, coord_min, coord_max, (0, 255, 0), 2)

        # Desenha o esqueleto
        joint_visible = detection_keypoints_score.flatten() > joint_threshold
        for joint0_idx, joint1_idx in JOINT_PAIRS:
            if joint_visible[joint0_idx] and joint_visible[joint1_idx]:
                p1 = scale_coord(detection_keypoints[joint0_idx])
                p2 = scale_coord(detection_keypoints[joint1_idx])
                cv2.line(image, p1, p2, (255, 0, 255), 3)

def draw_predictions(request):
    """
    Callback para desenhar as predições na janela de preview.
    """
    with MappedArray(request, 'main') as m:
        if last_predictions:
            visualize_pose_estimation_result(last_predictions, m.array, model_size)

# --- Lógica Principal de Execução ---

picam2 = Picamera2()
try:
    with Hailo(args.model) as hailo:
        # Define as resoluções dos streams de vídeo
        main_size = (1280, 720)
        model_h, model_w, _ = hailo.get_input_shape()
        model_size = lores_size = (model_w, model_h)

        # Configura a câmera
        config = picam2.create_video_configuration(
            main={'size': main_size, 'format': 'XRGB8888'},
            lores={'size': lores_size, 'format': 'RGB888'},
            controls={'FrameRate': 30}
        )
        picam2.configure(config)

        # Inicia a janela de preview
        picam2.start_preview(Preview.QTGL, x=0, y=0, width=main_size[0] // 2, height=main_size[1] // 2)
        
        # Configura o encoder e o buffer circular
        bitrate = 10000000  # 10 Mbps
        encoder = H264Encoder(bitrate=bitrate)
        
        # *** CORREÇÃO APLICADA AQUI ***
        # Calcula o tamanho do buffer em bytes para 20 segundos de vídeo
        seconds_to_buffer = 20
        buffer_size_bytes = int(bitrate / 8 * seconds_to_buffer)
        circular_output = CircularOutput(buffersize=buffer_size_bytes)
        
        picam2.start_recording(encoder, circular_output)
        
        # Define a função de callback para desenhar na tela
        picam2.pre_callback = draw_predictions
        
        print("🚀 Sistema iniciado. Aguardando detecção da pose...")

        while True:
            # Captura o frame de baixa resolução para a IA
            frame = picam2.capture_array('lores')
            
            # Executa a inferência com o Hailo
            raw_detections = hailo.run(frame)
            
            # Pós-processa os resultados
            last_predictions = postproc_yolov8_pose(1, raw_detections, model_size)

            # --- Lógica de Detecção e Gravação ---
            pose_found = False
            if last_predictions and not recording:
                scores, keypoints, joint_scores = (
                    last_predictions['scores'][0], last_predictions['keypoints'][0], last_predictions['joint_scores'][0])
                
                for i in range(len(scores)):
                    if scores[i][0] > 0.5:
                        if check_arms_crossed_above_head(keypoints[i], joint_scores[i].flatten()):
                            pose_found = True
                            break
            
            if pose_found and not recording:
                recording = True
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"gravacao_{timestamp}.h264"
                
                print(f"✅ Pose detectada! Salvando os últimos {seconds_to_buffer} segundos em {filename}")
                
                # *** LÓGICA DE GRAVAÇÃO CORRIGIDA ***
                # Atribui o nome do arquivo e inicia a escrita do buffer para o disco
                circular_output.fileoutput = filename
                circular_output.start()
                
                # Espera a gravação ser finalizada
                circular_output.stop()

                print(f"✅ Gravação '{filename}' finalizada. Sistema pronto para nova detecção.")
                recording = False

finally:
    # Garante que a câmera seja desligada corretamente
    if picam2.is_open:
        picam2.stop_recording()
    print("\nPrograma encerrado.")