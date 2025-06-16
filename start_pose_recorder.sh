#!/bin/bash

# Este script configura o ambiente e depois executa o gravador de pose.

# Garante que estamos no diretório correto
cd /home/d2snc/Documents/camera_test/

echo "Ativando ambiente Hailo para o gravador de pose..."
# Ativa o ambiente do Hailo
source /home/d2snc/Documents/hailo-rpi5-examples/setup_env.sh

echo "Iniciando o script pose_recorder.py..."
# Executa o script Python. Não precisa mais de 'sudo' por causa do Passo 1.
/usr/bin/python3 /home/d2snc/Documents/camera_test/pose_recorder.py