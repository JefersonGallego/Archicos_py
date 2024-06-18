import cv2
from pyzbar import pyzbar
import numpy as np
from keras.models import model_from_json
import tkinter as tk
from tkinter import StringVar
import threading
from datos import data2,clases_2

# Cargar el modelo desde el archivo JSON y los pesos desde el HDF5
with open("model.json", "r") as json_file:
    model_json = json_file.read()
    model = model_from_json(model_json)
model.load_weights("model.weights.h5")
print("Modelo y pesos cargados!")

# Compilar el modelo (necesario antes de hacer predicciones)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
# Definir las clases para decodificar después de la predicción

clases = clases_2

# Función para procesar la lectura de códigos de barras en tiempo real
def leer_codigos_de_barras():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error al abrir la cámara")
        return

    while not detener_hilo:
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar códigos de barras
        barcodes = pyzbar.decode(frame)

        for barcode in barcodes:
            # Obtener datos del código de barras
            barcodeData = barcode.data.decode("utf-8")

            # Preparar datos para la verificación del modelo
            max_len = len(barcodeData)  # Longitud máxima del número de serie (ajustado según necesidad)
            serial_padded = barcodeData.ljust(max_len)
            serial_ascii = np.array([ord(char) for char in serial_padded], dtype=np.float32) / 255.0
            serial_ascii = np.expand_dims(serial_ascii, axis=0)  # Añadir dimensión para predicción

            # Hacer la predicción con el modelo si hay datos válidos
            if len(serial_ascii) > 0:
                prediccion = model.predict(serial_ascii)
                clase_predicha = np.argmax(prediccion)
                producto_predicho = clases[clase_predicha]  # Usar el diccionario 'clases' para obtener la etiqueta

                # Actualizar las variables de Tkinter
                numero_serie_variable.set(barcodeData)
                tipo_producto_variable.set(producto_predicho)

                # Dibujar rectángulo alrededor del código de barras
                (x, y, w, h) = barcode.rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Mostrar el número de código de barras encima del rectángulo
                cv2.putText(frame, barcodeData, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Mostrar el producto predicho en la parte inferior izquierda del rectángulo
                cv2.putText(frame, "Producto: {}".format(producto_predicho), (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Mostrar el frame con los resultados
        cv2.imshow('Lectura de Código de Barras', frame)

        # Salir del bucle al presionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

# Crear la ventana principal de Tkinter
root = tk.Tk()
root.title("Lectura de Código de Barras")
root.geometry("400x200")

# Crear variables de control para el número de serie y el tipo de producto
numero_serie_variable = StringVar()
tipo_producto_variable = StringVar()

# Crear etiquetas enlazadas a las variables de control
etiqueta_numero_serie = tk.Label(root, text="Número de Serie:")
etiqueta_numero_serie.pack()

valor_numero_serie = tk.Label(root, textvariable=numero_serie_variable)
valor_numero_serie.pack()

etiqueta_tipo_producto = tk.Label(root, text="Tipo de Producto:")
etiqueta_tipo_producto.pack()

valor_tipo_producto = tk.Label(root, textvariable=tipo_producto_variable)
valor_tipo_producto.pack()

# Variable para detener el hilo
detener_hilo = False

# Crear un botón para iniciar la lectura de códigos de barras
def iniciar_lectura():
    global detener_hilo
    detener_hilo = False
    hilo = threading.Thread(target=leer_codigos_de_barras)
    hilo.start()

def detener_lectura():
    global detener_hilo
    detener_hilo = True

boton_iniciar = tk.Button(root, text="Iniciar Lectura", command=iniciar_lectura)
boton_iniciar.pack(pady=10)

boton_detener = tk.Button(root, text="Detener Lectura", command=detener_lectura)
boton_detener.pack(pady=10)

# Ejecutar el bucle principal de Tkinter
root.mainloop()
