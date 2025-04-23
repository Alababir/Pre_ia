import cv2
import numpy as np

# Intervalo HSV para bola preta
lower_black = np.array([0, 0, 0])
upper_black = np.array([179, 255, 83])

# Intervalo HSV para bola prateada (baixo tom de cor, brilho alto)
lower_silver = np.array([0, 0, 120])
upper_silver = np.array([179, 50, 255])

# Inicializa a captura da câmera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao abrir a câmera")
    exit()

# Definindo o intervalo de áreas para a bola preta
MIN_AREA = 500
MAX_AREA = 500000

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro na captura da câmera")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Máscaras de cor
    # Remove fundo branco da imagem
    mask_white = cv2.inRange(hsv, np.array([0, 0, 230]), np.array([179, 30, 255]))
    mask_silver = cv2.bitwise_and(mask_silver, cv2.bitwise_not(mask_white))


    # Detecta bola preta
    contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_black:
        area = cv2.contourArea(contour)
        if MIN_AREA < area < MAX_AREA:
            (x, y, w, h) = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, "Bola Preta", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

    # Detecta bola prateada com HoughCircles na máscara prateada
    blurred_silver = cv2.GaussianBlur(mask_silver, (15, 15), 0)
    circles_silver = cv2.HoughCircles(blurred_silver, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                      param1=50, param2=15, minRadius=10, maxRadius=50)

    if circles_silver is not None:
        circles_silver = np.round(circles_silver[0, :]).astype("int")
        for (x, y, r) in circles_silver:
            cv2.circle(frame, (x, y), r, (255, 0, 0), 2)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), 3)
            cv2.putText(frame, "Bola Prateada", (x - 40, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)

    # Mostra janelas com resultados
    cv2.imshow("Frame Original", frame)
    cv2.imshow("Mascara Preta", mask_black)
    cv2.imshow("Mascara Prata", mask_silver)

    # Tecla ESC para sair
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
