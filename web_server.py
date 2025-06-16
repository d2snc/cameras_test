#!/usr/bin/env python3

from flask import Flask, render_template, send_from_directory
import os

# Inicializa a aplicação Flask
app = Flask(__name__)

# Define o diretório onde os vídeos estão salvos (o diretório atual)
VIDEO_DIR = "."

@app.route('/')
def index():
    """
    Esta função é chamada quando alguém acessa a página principal.
    Ela lista os vídeos e renderiza o template HTML.
    """
    video_files = []
    # Itera sobre todos os arquivos no diretório
    for filename in os.listdir(VIDEO_DIR):
        # Adiciona à lista apenas os arquivos que terminam com .mp4
        if filename.endswith(".mp4"):
            video_files.append(filename)
    
    # Ordena a lista de vídeos para que os mais recentes apareçam primeiro
    video_files.sort(key=lambda x: os.path.getmtime(os.path.join(VIDEO_DIR, x)), reverse=True)
    
    # Renderiza a página 'index.html', passando a lista de nomes de arquivos de vídeo
    return render_template('index.html', videos=video_files)

@app.route('/videos/<path:filename>')
def serve_video(filename):
    """
    Esta função serve os arquivos de vídeo para o player na página web.
    """
    return send_from_directory(VIDEO_DIR, filename)

if __name__ == '__main__':
    print("🚀 Servidor Web iniciado! Acesse de outros dispositivos na sua rede.")
    print("Verifique o IP do seu Raspberry Pi com o comando 'hostname -I'")
    # O host '0.0.0.0' torna o servidor acessível por qualquer dispositivo na rede
    app.run(host='0.0.0.0', port=8000)