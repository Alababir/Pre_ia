import serial.tools.list_ports

def encontrar_arduino():
    portas = serial.tools.list_ports.comports()
    for porta in portas:
        if 'Arduino' in porta.description or 'ttyACM' in porta.device or 'ttyUSB' in porta.device:
            return porta.device
    return None

porta_arduino = encontrar_arduino()

if porta_arduino:
    print(f"Arduino encontrado na porta: {porta_arduino}")
    # Você pode abrir a porta agora:
    # ser = serial.Serial(porta_arduino, 9600)
else:
    print("Arduino não encontrado.")
