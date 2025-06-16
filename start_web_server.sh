#!/bin/bash

# Este script configura o ambiente e depois executa o servidor web.

# Garante que estamos no diretório correto
cd /home/d2snc/Documents/camera_test/

echo "Ativando ambiente Hailo para o servidor web..."
# Ativa o ambiente do Hailo (conforme solicitado)
source /home/d2snc/Documents/hailo-rpi5-examples/setup_env.sh

echo "Iniciando o script web_server.py..."
# Executa o script Python
/usr/bin/python3 /home/d2snc/Documents/camera_test/web_server.py