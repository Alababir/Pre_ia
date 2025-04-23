import cv2
import numpy as np

def nothing(x):
    pass

# Janela com sliders (trackbars)
cv2.namedWindow("Sliders HSV", cv2.WINDOW_NORMAL)

# Sliders para HSV Mínimo e Máximo
cv2.createTrackbar("H Min", "Sliders HSV", 0, 179, nothing)
cv2.createTrackbar("S Min", "Sliders HSV", 0, 255, nothing)
cv2.createTrackbar("V Min", "Sliders HSV", 0, 255, nothing)
cv2.createTrackbar("H Max", "Sliders HSV", 179, 179, nothing)
cv2.createTrackbar("S Max", "Sliders HSV", 255, 255, nothing)
cv2.createTrackbar("V Max", "Sliders HSV", 255, 255, nothing)

# Captura da webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Erro ao abrir a câmera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro na leitura da câmera")
        break

    # Conversão para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Lê os valores dos sliders
    h_min = cv2.getTrackbarPos("H Min", "Sliders HSV")
    s_min = cv2.getTrackbarPos("S Min", "Sliders HSV")
    v_min = cv2.getTrackbarPos("V Min", "Sliders HSV")
    h_max = cv2.getTrackbarPos("H Max", "Sliders HSV")
    s_max = cv2.getTrackbarPos("S Max", "Sliders HSV")
    v_max = cv2.getTrackbarPos("V Max", "Sliders HSV")

    # Aplica máscara HSV
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Mostra os resultados
    cv2.imshow("Original", frame)
    cv2.imshow("Mascara", mask)
    cv2.imshow("Resultado", result)

    # Sai com ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
