from ultralytics import YOLO
import cv2
import torch
import time
import sys
import termios
import tty
import select

# Função para capturar tecla pressionada sem bloquear
def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# Carregar o modelo
model = YOLO("piroka.pt")

# Parâmetros adicionais
conf_threshold = 0.2  # Limite de confiança
classes = [0, 1]  # Classes a serem detectadas
batch_size = 5  # Tamanho do batch

# Definir dispositivo para usar a GPU (se disponível)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

cap = cv2.VideoCapture(0)  # Qual câmera usar
camWidth = 160
camHeight = 90
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camHeight)

# Verificar se a câmera foi aberta corretamente
if not cap.isOpened():
    print("Erro ao abrir a câmera!")
    exit()

# Limitação da taxa de quadros (FPS)
frame_interval = 1 / 30  # fps
last_time = time.time()

# Lista para armazenar os frames do batch
batch_frames = []

print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar o frame")
        break

    # Limitar a taxa de quadros
    if time.time() - last_time < frame_interval:
        continue
    last_time = time.time()

    # Redimensionar o frame e adicionar ao batch
    frame_resized = cv2.resize(frame, (camWidth, camHeight))
    batch_frames.append(frame_resized)

    # Processar o batch quando atingir o tamanho definido
    if len(batch_frames) == batch_size:
        results = model.predict(source=batch_frames, conf=conf_threshold, classes=classes, show=False)

        # Processar resultados (exemplo: imprimir informações das detecções)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da caixa delimitadora
                conf = box.conf[0]  # Confiança da detecção
                cls = int(box.cls[0])  # Classe detectada
                print(f"Classe: {cls}, Confiança: {conf:.2f}, Caixa: ({x1}, {y1}, {x2}, {y2})")

        # Limpar o batch após o processamento
        batch_frames = []

    # Verificar se a tecla 'q' foi pressionada
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        if get_key() == 'q':
            break

# Liberar a captura e fechar janelas
cap.release()
cv2.destroyAllWindows()
