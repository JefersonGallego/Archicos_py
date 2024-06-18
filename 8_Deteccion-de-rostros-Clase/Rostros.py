# Importamos las librerias
import cv2

# Realizamos VideoCaptura
cap = cv2.VideoCapture(0)

# Arquitectura de Red Neuronal = opencv_face_detector.prototxt
# Pesos de Modelo Previamente Entrenado = res10_300x300_ssd_iter_140000.caffemodel

# Leemos el modelo
net = cv2.dnn.readNetFromCaffe("opencv_face_detector.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Parametros Modelo
# Tamaño el mismo con el que el Modelo se entreno 300x300
anchonet = 300
altonet = 300
# Valores medios de los canales de color
media = [104, 117, 123]
umbral = 0.7 # 70 % de Probabilidad

# Empezamos
while True:
    # Leer Frames
    ret, frame = cap.read()

    # Se leyeron Frames
    if not ret:
        break

    # Conversion tipo espejo espejo 
    frame = cv2.flip(frame, 1)

    # Extraer info Frames
    altoframe = frame.shape[0]
    anchoframe = frame.shape[1]

    # Preprocesamos la imagen
    # Images - Factor de escala - tamaño - media de color - Formato de color(BGR-RGB) - Recorte
    blob = cv2.dnn.blobFromImage(frame, 1.0, (anchonet, altonet), media, swapRB = False, crop = False)

    # Correr Modelo
    net.setInput(blob)
    detecciones = net.forward()

    # Iteramos
    for i in range(detecciones.shape[2]):
        # Extrar confianza de deteccion
        conf_detect = detecciones[0,0,i,2]
        # Confianza > Umbral 70%
        if conf_detect > umbral:
            # Extraemos las coordenadas
            xmin = int(detecciones[0, 0, i, 3] * anchoframe)
            ymin = int(detecciones[0, 0, i, 4] * altoframe)
            xmax = int(detecciones[0, 0, i, 5] * anchoframe)
            ymax = int(detecciones[0, 0, i, 6] * altoframe)

            # Dibujar rectangulo
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            # Texto
            label = "Confianza de deteccion: %.4f" % conf_detect
            # Tamaño del fondo del label
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Colocamos fondo al texto
            cv2.rectangle(frame, (xmin, ymin - label_size[1]), (xmin + label_size[0], ymin + base_line),
                          (0,0,0), cv2.FILLED)
            # Colocamps el texto
            cv2.putText(frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    cv2.imshow("DETECCION DE ROSTROS", frame)

    t = cv2.waitKey(1)
    if t == 27:
        break

cv2.destroyAllWindows()
cap.release()
