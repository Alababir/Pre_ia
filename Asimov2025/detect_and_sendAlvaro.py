import cv2
import numpy as np
import serial
import time
import glob

# Parâmetros da câmera
camWidth = 680
camHeight = 480
center_x = camWidth // 2

# HSV para a máscara de prata
silver_min = np.array([0, 0, 200])
silver_max = np.array([180, 40, 255])

# Variável da serial
ser = None

def conectar_serial():
    global ser
    ports = glob.glob('/dev/ttyACM*')
    for port in ports:
        try:
            ser = serial.Serial(port, 9600, timeout=1)
            time.sleep(2)
            print(f"[INFO] Conectado ao Arduino na porta {port}")
            return True
        except Exception as e:
            print(f"[ERRO] Falha ao conectar em {port}: {e}")

    ser = None
    return False

# Tenta conectar inicialmente
conectar_serial()

# Inicializa a câmera
cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.1, minDist=60,
                                   param1=25, param2=55, minRadius=10, maxRadius=100)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :1]:
                y1 = max(0, y - r)
                y2 = min(y + r, camHeight)
                x1 = max(0, x - r)
                x2 = min(x + r, camWidth)

                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                silver_mask = cv2.inRange(hsv_roi, silver_min, silver_max)

                if np.sum(silver_mask) > (silver_mask.size * 0.3):
                    error = x - center_x
                    data = f'0;{error}\n'

                    # Tenta enviar pela serial
                    if ser is not None:
                        try:
                            ser.write(data.encode('utf-8'))
                            print(f"[ENVIO] {data.strip()}")
                        except serial.SerialException as e:
                            print(f"[ERRO] Perdeu conexão com a serial: {e}")
                            ser = None

                    # Tenta reconectar se necessário
                    if ser is None:
                        conectar_serial()

                    # Desenho visual
                    cv2.circle(frame, (x, y), r, (192, 192, 192), 4)
                    cv2.line(frame, (x, y), (center_x, y), (255, 0, 0), 2)
                    cv2.putText(frame, f'Erro X: {error}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    break

        # Desenha centro da tela
        cv2.line(frame, (center_x, 0), (center_x, camHeight), (0, 0, 255), 2)
        cv2.imshow('Deteccao Bola Prata', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"[EXCEÇÃO] {e}")

finally:
    if ser is not None and ser.is_open:
        ser.close()
    cap.release()
    cv2.destroyAllWindows()
