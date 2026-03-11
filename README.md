# cameras_test

Repositório de configuração da câmera do XPLAYS

Dois scripts são fundamentais para o funcionamento da câmera, que é o pose_recorder.py e o web_server.py, sendo os dois executados através de um script que roda quando o rasp reinicia.

Para configurar para iniciar assim que o pc iniciar, coloque a seguinte configuração após dar o comando 'sudo crontab -e':

@reboot sh /home/d2snc/Documents/cameras_test/start_pose_recorder.sh > /home/d2snc/Documents/cameras_test/logs/cronlog 2>&1
@reboot sh /home/d2snc/Documents/cameras_test/start_web_server.sh > /home/d2snc/Documents/cameras_test/logs/cronlog 2>&1

Dessa forma o script já vai iniciar automaticamente assim que você reiniciar ai já pode começar a trabalhar
