import cv2
import numpy as np
import os

# Diretório com imagens de referência
DIR_REF = "imagens_referencia/"
arquivos = os.listdir(DIR_REF)

# Inicializa ORB
orb = cv2.ORB_create()

# Armazena descritores das imagens de referência
referencias = []
for arquivo in arquivos:
    img_ref = cv2.imread(os.path.join(DIR_REF, arquivo), 0)
    kp_ref, des_ref = orb.detectAndCompute(img_ref, None)
    referencias.append((img_ref, kp_ref, des_ref))

# Carrega a nova imagem
img_teste = cv2.imread("nova_imagem.jpg", 0)
kp_teste, des_teste = orb.detectAndCompute(img_teste, None)

# Inicializa o comparador de descritores
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

melhor_correspondencia = None
melhor_pontos = 0

for img_ref, kp_ref, des_ref in referencias:
    matches = bf.match(des_ref, des_teste)
    matches = sorted(matches, key=lambda x: x.distance)  # Ordena pelos melhores

    # Se houver correspondência suficiente, processa
    if len(matches) > melhor_pontos:
        melhor_pontos = len(matches)
        melhor_correspondencia = (img_ref, kp_ref, matches)

# Se encontrou algo
if melhor_correspondencia:
    img_ref, kp_ref, matches = melhor_correspondencia
    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_teste[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Encontra a transformação e a posição do objeto
    matriz, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if matriz is not None:
        altura, largura = img_ref.shape
        pts = np.float32([[0, 0], [largura, 0], [largura, altura], [0, altura]]).reshape(-1, 1, 2)
        destino = cv2.perspectiveTransform(pts, matriz)

        # Desenhar retângulo na nova imagem
        img_final = cv2.cvtColor(img_teste, cv2.COLOR_GRAY2BGR)
        cv2.polylines(img_final, [np.int32(destino)], True, (0, 255, 0), 3)
        
        cv2.imshow("Detecção", img_final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
