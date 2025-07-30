#!/usr/bin/env python3

from flask import Flask, render_template, send_from_directory, request, jsonify
import os
import json

# Inicializa a aplicação Flask
app = Flask(__name__)

# Define o diretório onde os vídeos estão salvos (o diretório atual)
VIDEO_DIR = "/home/d2snc/Documents/cameras_test"
DEBUG_CONFIG_FILE = os.path.join(VIDEO_DIR, "debug_config.json")

def load_debug_mode():
    """Carrega o estado atual do modo DEBUG"""
    try:
        if os.path.exists(DEBUG_CONFIG_FILE):
            with open(DEBUG_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config.get('debug_mode', False)
    except:
        pass
    return False

def save_debug_mode(debug_mode):
    """Salva o estado do modo DEBUG"""
    try:
        config = {'debug_mode': debug_mode}
        with open(DEBUG_CONFIG_FILE, 'w') as f:
            json.dump(config, f)
        return True
    except:
        return False

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
    
    # Carrega o estado atual do modo DEBUG
    debug_mode = load_debug_mode()
    
    # Renderiza a página 'index.html', passando a lista de nomes de arquivos de vídeo e o estado do DEBUG
    return render_template('index.html', videos=video_files, debug_mode=debug_mode)

@app.route('/videos/<path:filename>')
def serve_video(filename):
    """
    Esta função serve os arquivos de vídeo para o player na página web.
    """
    return send_from_directory(VIDEO_DIR, filename)

@app.route('/toggle_debug', methods=['POST'])
def toggle_debug():
    """
    Endpoint para alternar o estado do modo DEBUG
    """
    try:
        data = request.get_json()
        debug_mode = data.get('debug', False)
        
        if save_debug_mode(debug_mode):
            return jsonify({'debug': debug_mode, 'success': True})
        else:
            return jsonify({'error': 'Erro ao salvar configuração'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Servidor Web iniciado! Acesse de outros dispositivos na sua rede.")
    print("Verifique o IP do seu Raspberry Pi com o comando 'hostname -I'")
    # O host '0.0.0.0' torna o servidor acessível por qualquer dispositivo na rede
    app.run(host='0.0.0.0', port=8000)
