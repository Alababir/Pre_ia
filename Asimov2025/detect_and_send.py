from ultralytics import YOLO
import cv2
import torch
import serial

ser = serial.Serial('/dev/ttyACM0', 9600)  # Defina a porta serial correta e a velocidade (baudrate)

# Carregar o modelo
model = YOLO("best.pt")


# Parâmetros adicionais
conf_threshold = 0.6  # Limite de confiança
classes = [0,1]  # Classes a serem detectadas

# Inicializar a captura de vídeo da webcam
cap = cv2.VideoCapture(0)
camWidth = 680
camHeight = 480

while True:
    try:
        ret, frame = cap.read()
        frame=cv2.resize(frame,(camWidth,camHeight))
        if not ret:
            break

        # Fazer a predição no frame capturado
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            classes=classes,
            show=False,  # Não mostrar a imagem com OpenCV imshow, pois será feito manualmente
        )

        # Iterar sobre os resultados e desenhar caixas delimitadoras no frame original
        max_area=0
        max_box= None

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da caixa delimitadora
                area = (x2 - x1) * (y2 - y1)  # Área da caixa delimitadora
                
                if area > max_area:
                    max_area = area
                    max_box = box

        if max_box is not None:
            x1, y1, x2, y2 = map(int, max_box.xyxy[0])  # Coordenadas da caixa delimitadora
            conf = max_box.conf[0]  # Confiança da detecção
            label = f'{max_box.cls[0]} {conf:.2f}'  # Rótulo com a classe e confiança

        # Desenhar a caixa delimitadora e o rótulo
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(frame, ((x2+x1)//2,(y2+y1)//2),(camWidth//2,(y2+y1)//2),(255,0,0),2)
            error = ((x2+x1)//2 - (camWidth//2))

            info_to_send = (f'{max_box.cls[0]};{error} \n')
            
            ser.write(info_to_send.encode())

            cv2.line(frame,(camWidth//2,0),(camWidth//2,camHeight),(0,0,255),3)
            cv2.putText(frame, str(error), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.putText(frame, label, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


        # Mostrar o frame com as detecções
        cv2.imshow('Detections', frame)

    except KeyboardInterrupt:
        ser.close()
        cap.release()
        cv2.destroyAllWindows()

