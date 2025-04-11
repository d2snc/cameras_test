#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script Serial com MQTT:
- Abre a porta serial (por exemplo, /dev/ttyUSB0) e a lê continuamente.
- Quando dados são recebidos via serial, publica-os via MQTT (tópico "serial/incoming").
- Também escuta por mensagens MQTT (tópico "serial/outgoing") e, ao receber, escreve na porta serial.
"""

import serial
import time
import paho.mqtt.client as mqtt

# Configurações da porta serial
SERIAL_PORT = '/dev/ttyUSB0'
BAUDRATE = 115200

# Configurações do MQTT
MQTT_BROKER = "localhost"  # Altere se necessário
MQTT_PORT = 1883
TOPIC_INCOMING = "serial/incoming"   # Do serial para o main
TOPIC_OUTGOING = "serial/outgoing"    # Do main para o serial

# Tenta abrir a porta serial
try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    print("Porta serial aberta com sucesso.")
except Exception as e:
    print("Erro ao abrir porta serial:", e)
    ser = None

def on_connect(client, userdata, flags, rc):
    print("Conectado ao broker MQTT com código de resultado:", rc)
    client.subscribe(TOPIC_OUTGOING)

def on_message(client, userdata, msg):
    message = msg.payload.decode('utf-8').strip()
    print(f"MQTT - Mensagem recebida no tópico {msg.topic}: {message}")
    if ser is not None:
        try:
            ser.write(message.encode())
            print("Enviado comando para a porta serial:", message)
        except Exception as e:
            print("Erro ao escrever na porta serial:", e)
    else:
        print("Porta serial indisponível.")

mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

def read_serial():
    while True:
        if ser is not None and ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    print("Recebido da serial:", line)
                    mqtt_client.publish(TOPIC_INCOMING, line)
            except Exception as e:
                print("Erro ao ler da porta serial:", e)
        time.sleep(0.1)

try:
    read_serial()
except KeyboardInterrupt:
    print("Encerrando o script serial_mqtt.py...")
finally:
    if ser is not None:
        ser.close()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
