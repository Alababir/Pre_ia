import cv2
import numpy as np

# Intervalo HSV para bola preta
lower_black = np.array([0, 0, 0])
upper_black = np.array([179, 255, 83])

# Inicializa a captura da câmera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao abrir a câmera")
    exit()

# Definindo o intervalo de áreas para a bola preta
MIN_AREA = 500  # Área mínima para considerar a bola preta
MAX_AREA = 500000  # Área máxima para considerar a bola preta

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro na captura da câmera")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Máscara para detectar bola preta
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    # Detecção de círculos para a bola prateada (usando HoughCircles)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Detecta círculos usando a Transformada de Hough
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=30, minRadius=10, maxRadius=50)

    # Inicializa variáveis para detecção
    bola_preta_detectada = False
    bola_prata_detectada = False

    # Processa a bola preta
    contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_black:
        area = cv2.contourArea(contour)

        # Verifica se a área está dentro do intervalo definido
        if MIN_AREA < area < MAX_AREA:
            (x, y, w, h) = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, "Bola Preta Detectada", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            bola_preta_detectada = True  # Marca que a bola preta foi detectada
            break  # Sai do loop assim que uma bola preta for detectada

    # Se uma bola preta foi detectada, não detecta a bola prateada
    if not bola_preta_detectada:
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            # Processa a bola prateada, desenha o círculo médio
            if len(circles) > 0:
                # Se apenas um círculo for detectado, processa normalmente
                (x, y, r) = circles[0]
                cv2.circle(frame, (x, y), r, (255, 0, 0), 2)  # Desenha o círculo (bola prateada)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), 3)  # Desenha o centro do círculo
                cv2.putText(frame, "Bola Prateada Detectada", (x - 50, y - r - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                bola_prata_detectada = True  # Marca que a bola prateada foi detectada

    # Exibe a imagem original e a máscara preta
    cv2.imshow("Frame Original", frame)
    cv2.imshow("Mascara Preta", mask_black)

    # Aguarda a tecla ESC para sair
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
