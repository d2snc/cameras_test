#!/usr/bin/env python3
"""
Detector de braços cruzados acima da cabeça usando YOLOv8 Pose no Hailo 8L
Salva 20 segundos de vídeo anteriores ao gesto detectado
"""

import cv2
import numpy as np
from collections import deque
from datetime import datetime
import threading
import time
from picamera2 import Picamera2
from hailo_platform import (
    Device,
    HEF,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
    HailoStreamInterface,
    InferVStreams,
    InputVStreams,
    OutputVStreams
)

class PoseDetector:
    def __init__(self, hef_path, buffer_seconds=20, fps=30):
        """
        Inicializa o detector de pose
        
        Args:
            hef_path: Caminho para o arquivo .hef do YOLOv8 Pose
            buffer_seconds: Segundos de vídeo a manter no buffer
            fps: FPS do vídeo
        """
        self.hef_path = hef_path
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.frame_buffer = deque(maxlen=buffer_seconds * fps)
        self.saving_video = False
        self.detection_cooldown = 5  # Segundos entre detecções
        self.last_detection_time = 0
        
        # Índices dos keypoints do YOLOv8 Pose
        self.NOSE = 0
        self.LEFT_EYE = 1
        self.RIGHT_EYE = 2
        self.LEFT_EAR = 3
        self.RIGHT_EAR = 4
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_ELBOW = 7
        self.RIGHT_ELBOW = 8
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        
        # Inicializar Hailo
        self.setup_hailo()
        
        # Inicializar câmera
        self.setup_camera()
        
    def setup_hailo(self):
        """Configura o dispositivo Hailo"""
        # Carregar HEF
        self.hef = HEF(self.hef_path)
        
        # Obter dispositivos Hailo disponíveis
        # Para Hailo 8L no Raspberry Pi 5, geralmente é PCIe
        self.devices = Device.scan_pcie()
        if not self.devices:
            raise RuntimeError("Nenhum dispositivo Hailo encontrado!")
        
        self.device = self.devices[0]
        print(f"Usando dispositivo Hailo: {self.device}")
        
        # Configurar parâmetros com a interface correta
        self.target = ConfigureParams.create_from_hef(
            self.hef, 
            interface=HailoStreamInterface.PCIe
        )
        
        # Configurar parâmetros de entrada
        self.input_vstreams_params = InputVStreamParams.make(
            self.hef, 
            format_type=FormatType.FLOAT32
        )
        
        # Configurar parâmetros de saída
        self.output_vstreams_params = OutputVStreamParams.make(
            self.hef,
            format_type=FormatType.FLOAT32
        )
        
        # Criar interface de streaming
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()[0]
        
        # Configurar rede
        self.configured_infer_model = self.device.configure(
            self.hef,
            self.target
        )
        self.network_name = self.hef.get_network_group_names()[0]
        
    def setup_camera(self):
        """Configura a PiCamera2"""
        self.camera = Picamera2()
        
        # Configurar para 640x640 (entrada padrão do YOLO)
        config = self.camera.create_preview_configuration(
            main={"size": (640, 640), "format": "RGB888"},
            controls={"FrameRate": self.fps}
        )
        self.camera.configure(config)
        self.camera.start()
        
    def preprocess_frame(self, frame):
        """Preprocessa o frame para o modelo"""
        # Redimensionar se necessário
        if frame.shape[:2] != (640, 640):
            frame = cv2.resize(frame, (640, 640))
        
        # Normalizar para [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Adicionar dimensão do batch
        frame = np.expand_dims(frame, axis=0)
        
        return frame
    
    def postprocess_output(self, output):
        """Processa a saída do modelo para extrair poses"""
        # A saída do YOLOv8 Pose geralmente tem formato:
        # [batch, num_detections, 56] onde 56 = 4 (bbox) + 1 (conf) + 17*3 (keypoints)
        
        poses = []
        for detection in output[0]:
            confidence = detection[4]
            
            if confidence > 0.5:  # Threshold de confiança
                # Extrair keypoints (17 pontos x 3 valores: x, y, confiança)
                keypoints = detection[5:].reshape(17, 3)
                poses.append({
                    'bbox': detection[:4],
                    'confidence': confidence,
                    'keypoints': keypoints
                })
        
        return poses
    
    def check_arms_crossed_overhead(self, keypoints):
        """
        Verifica se os braços estão cruzados acima da cabeça
        
        Args:
            keypoints: Array de keypoints (17x3)
            
        Returns:
            bool: True se os braços estão cruzados acima da cabeça
        """
        # Verificar se os keypoints necessários têm confiança suficiente
        min_confidence = 0.3
        
        # Pontos necessários
        left_wrist = keypoints[self.LEFT_WRIST]
        right_wrist = keypoints[self.RIGHT_WRIST]
        left_shoulder = keypoints[self.LEFT_SHOULDER]
        right_shoulder = keypoints[self.RIGHT_SHOULDER]
        nose = keypoints[self.NOSE]
        
        # Verificar confiança
        if (left_wrist[2] < min_confidence or 
            right_wrist[2] < min_confidence or
            left_shoulder[2] < min_confidence or
            right_shoulder[2] < min_confidence or
            nose[2] < min_confidence):
            return False
        
        # Verificar se os pulsos estão acima da cabeça
        head_y = nose[1]
        if left_wrist[1] > head_y or right_wrist[1] > head_y:
            return False
        
        # Verificar se os braços estão cruzados
        # (pulso esquerdo à direita do ombro direito e vice-versa)
        if (left_wrist[0] > right_shoulder[0] and 
            right_wrist[0] < left_shoulder[0]):
            return True
        
        return False
    
    def save_video_buffer(self):
        """Salva o buffer de vídeo em um arquivo"""
        if self.saving_video:
            return
        
        self.saving_video = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"arms_crossed_{timestamp}.mp4"
        
        # Criar cópia do buffer para não interferir na captura
        frames_to_save = list(self.frame_buffer)
        
        # Salvar em thread separada
        def save_thread():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, self.fps, (640, 640))
            
            for frame in frames_to_save:
                # Converter RGB para BGR para o OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"Vídeo salvo: {filename}")
            self.saving_video = False
        
        threading.Thread(target=save_thread).start()
    
    def run_inference(self, frame):
        """Executa inferência no frame"""
        # Preprocessar
        input_data = self.preprocess_frame(frame)
        
        # Criar bindings
        with InferVStreams(self.configured_infer_model, self.input_vstreams_params, 
                          self.output_vstreams_params) as infer_pipeline:
            
            # Preparar dados de entrada
            input_dict = {self.input_vstream_info.name: input_data}
            
            # Executar inferência
            with self.configured_infer_model.activate(self.network_name):
                output = infer_pipeline.infer(input_dict)
                
        # Processar saída
        output_array = output[self.output_vstream_info.name]
        poses = self.postprocess_output(output_array)
        
        return poses
    
    def draw_pose(self, frame, poses):
        """Desenha as poses detectadas no frame"""
        for pose in poses:
            keypoints = pose['keypoints']
            
            # Desenhar conexões do esqueleto
            connections = [
                (5, 6),   # ombros
                (5, 7),   # ombro esquerdo - cotovelo
                (7, 9),   # cotovelo esquerdo - pulso
                (6, 8),   # ombro direito - cotovelo
                (8, 10),  # cotovelo direito - pulso
            ]
            
            for connection in connections:
                pt1_idx, pt2_idx = connection
                pt1 = keypoints[pt1_idx]
                pt2 = keypoints[pt2_idx]
                
                if pt1[2] > 0.3 and pt2[2] > 0.3:
                    cv2.line(frame, 
                            (int(pt1[0] * 640), int(pt1[1] * 640)),
                            (int(pt2[0] * 640), int(pt2[1] * 640)),
                            (0, 255, 0), 2)
            
            # Desenhar keypoints
            for i, kpt in enumerate(keypoints):
                if kpt[2] > 0.3:
                    cv2.circle(frame, 
                              (int(kpt[0] * 640), int(kpt[1] * 640)), 
                              3, (0, 0, 255), -1)
        
        return frame
    
    def run(self):
        """Loop principal do detector"""
        print("Iniciando detecção de braços cruzados...")
        print(f"Buffer de vídeo: {self.buffer_seconds} segundos")
        print("Pressione 'q' para sair")
        
        try:
            while True:
                # Capturar frame
                frame = self.camera.capture_array()
                
                # Adicionar ao buffer
                self.frame_buffer.append(frame.copy())
                
                # Executar inferência
                poses = self.run_inference(frame)
                
                # Verificar cada pose detectada
                current_time = time.time()
                for pose in poses:
                    if self.check_arms_crossed_overhead(pose['keypoints']):
                        # Verificar cooldown
                        if current_time - self.last_detection_time > self.detection_cooldown:
                            print("Braços cruzados detectados! Salvando vídeo...")
                            self.save_video_buffer()
                            self.last_detection_time = current_time
                
                # Desenhar poses para visualização
                display_frame = self.draw_pose(frame.copy(), poses)
                
                # Converter RGB para BGR para exibição
                display_frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                
                # Adicionar texto de status
                status = "Gravando..." if self.saving_video else "Monitorando"
                cv2.putText(display_frame_bgr, status, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Mostrar frame
                cv2.imshow('Pose Detection', display_frame_bgr)
                
                # Verificar saída
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nEncerrando...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpa recursos"""
        self.camera.stop()
        cv2.destroyAllWindows()
        print("Detector encerrado.")

if __name__ == "__main__":
    # Caminho para o arquivo HEF
    HEF_PATH = "yolov8s_pose.hef"
    
    # Criar e executar detector
    detector = PoseDetector(HEF_PATH, buffer_seconds=20, fps=30)
    detector.run()