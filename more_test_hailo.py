#!/usr/bin/env python3
"""
Script para testar a instalação e funcionamento do Hailo 8L
"""

import sys
import numpy as np

def test_hailo_import():
    """Testa se os módulos Hailo podem ser importados"""
    print("1. Testando importação dos módulos Hailo...")
    try:
        from hailo_platform import Device, HEF, VDevice
        print("✓ Módulos Hailo importados com sucesso")
        return True
    except ImportError as e:
        print(f"✗ Erro ao importar módulos Hailo: {e}")
        print("\nDica: Verifique se o HailoRT está instalado corretamente")
        print("Execute: pip install hailo_platform")
        return False

def test_device_detection():
    """Testa se o dispositivo Hailo é detectado"""
    print("\n2. Procurando dispositivos Hailo...")
    try:
        from hailo_platform import Device
        
        # Tentar diferentes métodos de detecção
        devices = []
        
        # Método 1: PCIe scan
        try:
            pcie_devices = Device.scan_pcie()
            devices.extend(pcie_devices)
            print(f"✓ Dispositivos PCIe encontrados: {len(pcie_devices)}")
        except:
            print("- Nenhum dispositivo PCIe encontrado")
        
        # Método 2: Scan geral
        try:
            all_devices = Device.scan()
            devices.extend(all_devices)
            print(f"✓ Total de dispositivos encontrados: {len(all_devices)}")
        except:
            pass
        
        if devices:
            print("\nDispositivos Hailo detectados:")
            for i, device in enumerate(devices):
                print(f"  Dispositivo {i}: {device}")
            return True
        else:
            print("✗ Nenhum dispositivo Hailo encontrado")
            print("\nDicas:")
            print("1. Verifique se o Hailo 8L está instalado corretamente")
            print("2. Execute: lspci | grep Hailo")
            print("3. Reinicie o Raspberry Pi se necessário")
            return False
            
    except Exception as e:
        print(f"✗ Erro ao procurar dispositivos: {e}")
        return False

def test_hef_loading(hef_path):
    """Testa o carregamento do arquivo HEF"""
    print(f"\n3. Testando carregamento do HEF: {hef_path}")
    try:
        from hailo_platform import HEF
        
        hef = HEF(hef_path)
        print("✓ Arquivo HEF carregado com sucesso")
        
        # Obter informações do modelo
        print("\nInformações do modelo:")
        print(f"  Redes: {hef.get_network_group_names()}")
        
        # Informações de entrada
        input_infos = hef.get_input_vstream_infos()
        print(f"\n  Entradas ({len(input_infos)}):")
        for info in input_infos:
            print(f"    - {info.name}: {info.shape}")
        
        # Informações de saída
        output_infos = hef.get_output_vstream_infos()
        print(f"\n  Saídas ({len(output_infos)}):")
        for info in output_infos:
            print(f"    - {info.name}: {info.shape}")
            
        return True
        
    except FileNotFoundError:
        print(f"✗ Arquivo não encontrado: {hef_path}")
        return False
    except Exception as e:
        print(f"✗ Erro ao carregar HEF: {e}")
        return False

def test_inference_setup(hef_path):
    """Testa a configuração para inferência"""
    print(f"\n4. Testando configuração de inferência...")
    try:
        from hailo_platform import (
            Device, HEF, VDevice, 
            HailoStreamInterface, ConfigureParams
        )
        
        # Carregar HEF
        hef = HEF(hef_path)
        
        # Criar VDevice
        params = VDevice.create_params()
        params.device_count = 1
        
        with VDevice(params) as vdevice:
            print("✓ VDevice criado com sucesso")
            
            # Configurar
            configure_params = ConfigureParams.create_from_hef(
                hef=hef,
                interface=HailoStreamInterface.PCIe
            )
            
            network_name = hef.get_network_group_names()[0]
            configured = vdevice.configure(hef, configure_params)
            
            print(f"✓ Rede '{network_name}' configurada com sucesso")
            
            # Testar ativação
            network_group = configured[network_name]
            with network_group.activate():
                print("✓ Rede ativada com sucesso")
            
        return True
        
    except Exception as e:
        print(f"✗ Erro na configuração: {e}")
        print(f"\nDetalhes: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_camera():
    """Testa a PiCamera2"""
    print("\n5. Testando câmera...")
    try:
        from picamera2 import Picamera2
        
        camera = Picamera2()
        config = camera.create_preview_configuration(
            main={"size": (640, 640), "format": "RGB888"}
        )
        camera.configure(config)
        camera.start()
        
        # Capturar um frame de teste
        frame = camera.capture_array()
        camera.stop()
        
        print(f"✓ Câmera funcionando: frame capturado {frame.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Erro com a câmera: {e}")
        print("\nDica: Execute 'sudo raspi-config' e ative a câmera")
        return False

def main():
    """Executa todos os testes"""
    print("=== Teste de Configuração Hailo 8L ===\n")
    
    # Verificar argumentos
    hef_path = "yolov8s_pose.hef"
    if len(sys.argv) > 1:
        hef_path = sys.argv[1]
    
    # Executar testes
    tests = [
        ("Importação", test_hailo_import()),
    ]
    
    if tests[0][1]:  # Se importação funcionou
        tests.extend([
            ("Detecção de dispositivo", test_device_detection()),
            ("Carregamento HEF", test_hef_loading(hef_path)),
            ("Configuração de inferência", test_inference_setup(hef_path)),
            ("Câmera", test_camera()),
        ])
    
    # Resumo
    print("\n=== Resumo dos Testes ===")
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for name, result in tests:
        status = "✓ Passou" if result else "✗ Falhou"
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed}/{total} testes passaram")
    
    if passed == total:
        print("\n✓ Sistema pronto para uso!")
    else:
        print("\n✗ Alguns problemas foram encontrados. Verifique os erros acima.")

if __name__ == "__main__":
    main()