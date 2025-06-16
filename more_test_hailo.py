#!/usr/bin/env python3
"""
Versão simplificada do detector usando HailoRT Runner API
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
    VDevice,
    HailoStreamInterface,
    ConfigureParams,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    QuantizationParam,
    FormatType
)

class SimplePoseDetector:
    def __init__(self, hef_path, buffer_seconds=20, fps=30):
        self.hef_path = hef_path
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.frame_buffer = deque(maxlen=buffer_seconds * fps)
        self.saving_video = False
        self.detection_cooldown = 5
        self.last_detection_time = 0
        
        # Keypoints indices
        self.NOSE = 0
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        
        self.setup_hailo_simple()
        self.setup_camera()
    
    def setup_hailo_simple(self):
        """Setup Hailo usando a API simplificada"""
        # Criar Virtual Device (VDevice)
        params = VDevice.create_params()
        params.device_count = 1
        
        with VDevice(params) as vdevice:
            # Carregar HEF
            hef = HEF(self.hef_path)
            
            # Obter informações da rede
            network_name = hef.get_network_group_names()[0]
            network_group = hef.get_network_group(network_name)
            
            # Configurar dispositivo
            configure_params = ConfigureParams.create_from_hef(
                hef=hef,
                interface=HailoStreamInterface.PCIe
            )
            
            network_group_handle = vdevice.configure(hef, configure_params)[network_name]
            
            # Salvar handles para uso posterior
            self.vdevice = vdevice
            self.hef = hef
            self.network_group_handle = network_group_handle
            self.network_name = network_name
            
            # Obter informações de entrada/saída
            self.input_vstream_info = hef.get_input_vstream_infos()[0]
            self.output_vstream_info = hef.get_output_vstream_infos()[0]
            
            print(f"Modelo carregado: {network_name}")
            print(f"Entrada: {self.input_vstream_info.shape}")
            print(f"Saída: {self.output_vstream_info.shape}")
    
    def setup_camera(self):
        """Configura a PiCamera2"""
        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(
            main={"size": (640, 640), "format": "RGB888"},
            controls={"FrameRate": self.fps}
        )
        self.camera.configure(config)
        self.camera.start()
    
    def preprocess_frame(self, frame):
        """Preprocessa o frame"""
        # Redimensionar se necessário
        input_shape = self.input_vstream_info.shape
        if frame.shape[:2] != (input_shape[1], input_shape[2]):
            frame = cv2.resize(frame, (input_shape[2], input_shape[1]))
        
        # Converter para o formato esperado
        frame = frame.astype(np.float32) / 255.0
        
        # Transpor se necessário (NHWC -> NCHW)
        if len(input_shape) == 4 and input_shape[1] == 3:
            frame = np.transpose(frame, (2, 0, 1))
        
        # Adicionar batch dimension
        frame = np.expand_dims(frame, axis=0)
        
        return frame
    
    def run_inference_simple(self, frame):
        """Executa inferência de forma simplificada"""
        input_data = self.preprocess_frame(frame)
        
        # Preparar buffers
        input_buffer = {self.input_vstream_info.name: input_data}
        output_buffer = {self.output_vstream_info.name: np.empty(self.output_vstream_info.shape)}
        
        # Executar inferência
        with self.network_group_handle.activate():
            self.network_group_handle.wait_for_activation(100)
            
            # Enviar dados
            self.network_group_handle.send_input_frame(
                self.input_vstream_info.name,
                input_data
            )
            
            # Receber resultado
            output = self.network_group_handle.recv_output_frame(
                self.output_vstream_info.name
            )
        
        return self.postprocess_output(output)
    
    def postprocess_output(self, output):
        """Processa saída do modelo"""
        poses = []
        
        # Assumindo formato YOLOv8 Pose
        # output shape: [1, num_detections, 56]
        detections = output[0]
        
        for detection in detections:
            confidence = detection[4]
            if confidence > 0.5:
                keypoints = detection[5:].reshape(17, 3)
                poses.append({
                    'keypoints': keypoints,
                    'confidence': confidence
                })
        
        return poses
    
    def check_arms_crossed(self, keypoints):
        """Verifica braços cruzados acima da cabeça"""
        # Extrair pontos relevantes
        nose = keypoints[self.NOSE]
        left_wrist = keypoints[self.LEFT_WRIST]
        right_wrist = keypoints[self.RIGHT_WRIST]
        left_shoulder = keypoints[self.LEFT_SHOULDER]
        right_shoulder = keypoints[self.RIGHT_SHOULDER]
        
        # Verificar confiança mínima
        if any(kp[2] < 0.3 for kp in [nose, left_wrist, right_wrist, left_shoulder, right_shoulder]):
            return False
        
        # Pulsos acima da cabeça?
        if left_wrist[1] > nose[1] or right_wrist[1] > nose[1]:
            return False
        
        # Braços cruzados?
        if left_wrist[0] > right_shoulder[0] and right_wrist[0] < left_shoulder[0]:
            return True
        
        return False
    
    def save_video(self):
        """Salva buffer de vídeo"""
        if self.saving_video:
            return
        
        self.saving_video = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"arms_crossed_{timestamp}.mp4"
        
        frames = list(self.frame_buffer)
        
        def save_thread():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, self.fps, (640, 640))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"Vídeo salvo: {filename}")
            self.saving_video = False
        
        threading.Thread(target=save_thread).start()
    
    def run(self):
        """Loop principal"""
        print("Detector iniciado. Pressione 'q' para sair.")
        
        try:
            while True:
                # Capturar frame
                frame = self.camera.capture_array()
                self.frame_buffer.append(frame.copy())
                
                # Inferência
                poses = self.run_inference_simple(frame)
                
                # Verificar poses
                current_time = time.time()
                for pose in poses:
                    if self.check_arms_crossed(pose['keypoints']):
                        if current_time - self.last_detection_time > self.detection_cooldown:
                            print("Braços cruzados detectados!")
                            self.save_video()
                            self.last_detection_time = current_time
                
                # Visualização simples
                display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                status = "Gravando..." if self.saving_video else "Monitorando"
                cv2.putText(display, status, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Pose Detection', display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
            print("Detector encerrado.")

if __name__ == "__main__":
    detector = SimplePoseDetector("yolov8s_pose.hef")
    detector.run()