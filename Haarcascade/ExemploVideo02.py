import cv2

#webCamera = cv2.VideoCapture('http://10.134.154.103')
webCamera = cv2.VideoCapture(0)
classificadorVideoFace = cv2.CascadeClassifier('cascade.xml')

while True:
    camera, frame = webCamera.read()

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detecta = classificadorVideoFace.detectMultiScale(cinza, scaleFactor=1.5, minSize=(150, 150))
    #detecta = classificadorVideoFace.detectMultiScale(cinza)
    for(x, y, l, a) in detecta:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (255, 0, 0), 2)

    cv2.imshow("Video WebCamera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

webCamera.release()
cv2.destroyAllWindows()
