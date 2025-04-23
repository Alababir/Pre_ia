import cv2
import numpy as np

# Intervalos HSV
lower_black = np.array([0, 0, 0])
upper_black = np.array([179, 255, 83])

lower_silver = np.array([0, 0, 120])
upper_silver = np.array([179, 50, 255])

# Fundo branco em HSV (para remover da detecção prata)
lower_white = np.array([0, 0, 230])
upper_white = np.array([179, 30, 255])

# Inicializa a câmera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao abrir a câmera")
    exit()

MIN_AREA = 500
MAX_AREA = 500000

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro na captura da câmera")
        break

    # Aplica CLAHE para melhorar contraste
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    frame_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(frame_clahe, cv2.COLOR_BGR2HSV)

    # Máscara da bola preta
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # Máscara original da prata
    mask_silver_raw = cv2.inRange(hsv, lower_silver, upper_silver)

    # Máscara do fundo branco
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Remove o fundo branco da máscara da prata
    mask_silver = cv2.bitwise_and(mask_silver_raw, cv2.bitwise_not(mask_white))

    # ========== DETECÇÃO BOLA PRETA ==========
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

    # ========== DETECÇÃO BOLA PRATEADA ==========
    blurred_silver = cv2.GaussianBlur(mask_silver, (15, 15), 0)
    circles_silver = cv2.HoughCircles(blurred_silver, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                      param1=50, param2=15, minRadius=10, maxRadius=50)

    if circles_silver is not None:
        circles_silver = np.round(circles_silver[0, :]).astype("int")
        for (x, y, r) in circles_silver:
            cv2.circle(frame, (x, y), r, (255, 0, 0), 2)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), 3)
            cv2.putText(frame, "Bola Prateada", (x - 40, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # ========== EXIBIÇÃO ==========
    cv2.imshow("Frame Original", frame)
    cv2.imshow("Mascara Preta", mask_black)
    cv2.imshow("Mascara Prata (original)", mask_silver_raw)
    cv2.imshow("Mascara Prata (fundo removido)", mask_silver)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
