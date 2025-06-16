#!/usr/bin/env python3
"""
Script de teste para fazer uma inferência básica no Hailo
"""

import numpy as np
from hailo_platform import HEF, VDevice, ConfigureParams, HailoStreamInterface

def test_basic_inference(hef_path):
    """Testa inferência básica com o Hailo"""
    print("=== Teste de Inferência Básica ===\n")
    
    try:
        # 1. Carregar HEF
        print("1. Carregando HEF...")
        hef = HEF(hef_path)
        print("✓ HEF carregado")
        
        # 2. Obter informações
        network_names = hef.get_network_group_names()
        print(f"\n2. Redes: {network_names}")
        
        input_infos = hef.get_input_vstream_infos()
        output_infos = hef.get_output_vstream_infos()
        
        print(f"\nEntrada:")
        for info in input_infos:
            print(f"  - {info.name}: {info.shape}")
            
        print(f"\nSaídas ({len(output_infos)}):")
        for i, info in enumerate(output_infos):
            print(f"  {i}: {info.name}: {info.shape}")
            
        # 3. Criar dados de teste
        print("\n3. Criando dados de teste...")
        input_shape = input_infos[0].shape
        # Criar imagem aleatória RGB
        test_image = np.random.randint(0, 255, size=(input_shape[0], input_shape[1], input_shape[2]), dtype=np.uint8)
        print(f"✓ Imagem de teste criada: {test_image.shape}")
        
        # 4. Criar VDevice
        print("\n4. Criando VDevice...")
        params = VDevice.create_params()
        params.device_count = 1
        
        with VDevice(params) as vdevice:
            print("✓ VDevice criado")
            
            # 5. Configurar
            print("\n5. Configurando rede...")
            configure_params = ConfigureParams.create_from_hef(
                hef, 
                interface=HailoStreamInterface.PCIe
            )
            
            # Configure
            network_group = vdevice.configure(hef, configure_params)
            print(f"✓ Configurado. Tipo retornado: {type(network_group)}")
            
            # Se for dicionário, pegar o primeiro network group
            if isinstance(network_group, dict):
                network_name = list(network_group.keys())[0]
                network_group = network_group[network_name]
                print(f"✓ Usando network group: {network_name}")
            
            # 6. Criar bindings
            print("\n6. Criando bindings...")
            bindings = network_group.create_bindings()
            print("✓ Bindings criados")
            
            # 7. Configurar entrada
            print("\n7. Configurando buffers...")
            input_name = input_infos[0].name
            bindings.input(input_name).set_buffer(test_image)
            print(f"✓ Buffer de entrada configurado: {input_name}")
            
            # 8. Configurar saídas
            output_buffers = {}
            for i, output_info in enumerate(output_infos):
                shape = list(output_info.shape)
                buffer = np.empty(shape, dtype=np.float32)
                bindings.output(output_info.name).set_buffer(buffer)
                output_buffers[output_info.name] = buffer
                print(f"✓ Buffer de saída {i} configurado: {output_info.name}")
                
            # 9. Executar inferência
            print("\n8. Executando inferência...")
            with network_group.activate(bindings):
                job = network_group.run_async(bindings, None)
                status = job.wait(5000)  # 5 segundos timeout
                print(f"✓ Inferência completa. Status: {status}")
                
            # 10. Verificar resultados
            print("\n9. Resultados:")
            for name, buffer in output_buffers.items():
                print(f"  - {name}: shape={buffer.shape}, min={buffer.min():.3f}, max={buffer.max():.3f}, mean={buffer.mean():.3f}")
                
        print("\n✓ Teste completo com sucesso!")
        return True
        
    except Exception as e:
        print(f"\n✗ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_activation(hef_path):
    """Teste ainda mais simples"""
    print("\n=== Teste Simples de Ativação ===\n")
    
    try:
        hef = HEF(hef_path)
        
        # Parâmetros mínimos
        vdevice_params = VDevice.create_params()
        vdevice_params.device_count = 1
        
        # Criar VDevice
        vdevice = VDevice(vdevice_params)
        print("✓ VDevice criado")
        
        # Configurar
        params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_groups = vdevice.configure(hef, params)
        print(f"✓ Configurado. Tipo: {type(network_groups)}")
        
        # Listar conteúdo
        if isinstance(network_groups, dict):
            print(f"  Chaves: {list(network_groups.keys())}")
        elif hasattr(network_groups, '__len__'):
            print(f"  Tamanho: {len(network_groups)}")
        
        # Limpar
        vdevice.release()
        print("✓ VDevice liberado")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python3 test_inference.py <arquivo.hef>")
        sys.exit(1)
        
    hef_path = sys.argv[1]
    
    # Teste simples primeiro
    if test_simple_activation(hef_path):
        print("\n" + "="*50 + "\n")
        # Teste completo
        test_basic_inference(hef_path)
    else:
        print("\nTeste simples falhou. Verifique a instalação do Hailo.")