from ultralytics import YOLO
import cv2
import serial

ser = serial.Serial('/dev/ttyACM0', 9600)  # Defina a porta serial correta e a velocidade (baudrate)


model = YOLO("piroka.pt")

# Parâmetros adicionais
conf_threshold = 0.6  # Limite de confiança
classes = [0, 1]  # Classes a serem detectadas
batch_size = 4 # Tamanho do batch


cap = cv2.VideoCapture(0)
camWidth = 240
camHeight = 180

# Variável de controle para exibir ou não as detecções
show_detections = True  

# Armazenar os frames para processamento em batch
batch_frames = []

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar o frame
        frame = cv2.resize(frame, (camWidth, camHeight))
        batch_frames.append(frame)

        # Processar o batch de frames
        if len(batch_frames) == batch_size:
            results = model.predict(source=batch_frames, conf=conf_threshold, classes=classes, show=False)

            # Encontrar a caixa com maior confiança no lote
            highest_conf = 0
            best_box = None
            best_frame_idx = None

            for frame_idx, result in enumerate(results):
                for box in result.boxes:
                    conf = box.conf[0]  # Confiança da detecção
                    if conf > highest_conf:
                        highest_conf = conf
                        best_box = box
                        best_frame_idx = frame_idx

            if best_box is not None:
                # Obter as coordenadas da melhor caixa
                coords = best_box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)
                conf = best_box.conf[0]
                label = f'{best_box.cls[0]} {conf:.2f}'

                # Calcular centro da caixa e erro em relação ao centro da câmera
                box_center_x = (x1 + x2) // 2
                box_center_y = (y1 + y2) // 2
                cam_center_x = camWidth // 2
                cam_center_y = camHeight // 2
                error_x = box_center_x - cam_center_x
                error_y = box_center_y - cam_center_y

                print(f"Erro: ({error_x}) e Classe ({best_box.cls})")

                # Desenhar no frame correspondente
                if show_detections:
                    frame_to_draw = batch_frames[best_frame_idx]
                    cv2.rectangle(frame_to_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.line(frame_to_draw, (box_center_x, box_center_y), (cam_center_x, box_center_y), (255, 0, 0), 2)
                    cv2.line(frame_to_draw, (cam_center_x, cam_center_y), (cam_center_x, box_center_y), (0, 255, 0), 2)

                    # Exibir informações no frame
                    cv2.putText(frame_to_draw, f'Coords: ({x1},{y1}), ({x2},{y2})', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame_to_draw, f'Error X: {error_x}', (box_center_x + 10, box_center_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(frame_to_draw, f'Error Y: {error_y}', (cam_center_x + 10, box_center_y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame_to_draw, label, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    

                    # Mostrar o frame com detecções
                    cv2.imshow('Detections', frame_to_draw)
                    

                # Enviar informações pela serial
                info_to_send = f'{best_box.cls[0]};{error_x};{error_y}\n'
                ser.write(info_to_send.encode())

            # Limpar o lote de frames
            batch_frames = []

        # Fechar a janela se a tecla 'q' for pressionada
        if show_detections and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        break

# Liberar recursos
ser.close()
cap.release()
cv2.destroyAllWindows()
