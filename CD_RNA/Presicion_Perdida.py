import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Datos de ejemplo
datos = [
    ("7702439001070", "A"),
    ("7700304444625", "B"),
    ("7700304387687", "C"),
    ("7700304443567", "D"),
    ("7700243000028", "E"),
    ("8719200268203", "F")
]

# Convertir números de serie a representación numérica (por ejemplo, longitud del número)
X_data = np.array([len(numero_serial) for numero_serial, clase in datos], dtype=np.float32)

# Codificación one-hot para las clases
clases = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
y_data = np.array([clases[clase] for numero_serial, clase in datos], dtype=np.int32)

# Normalización opcional de los datos de entrada (en este caso, la longitud del número de serie)
X_data = X_data / np.max(X_data)

# Inicio arquitectura de red neuronal 
model = Sequential()
model.add(Dense(8, input_dim=1, activation='relu'))    # Capa Oculta 1, activación ReLU
model.add(Dense(6, activation='softmax'))              # Capa de Salida, activación Softmax para clasificación multiclase

# Argumentos para el Aprendizaje
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Entrenamiento red neuronal
history = model.fit(X_data, y_data, epochs=100, batch_size=1, verbose=1)

# Evaluación del modelo
scores = model.evaluate(X_data, y_data)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Visualización del entrenamiento
plt.figure(figsize=(10, 6))

# Precisión
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')

# Pérdida
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')

plt.tight_layout()
plt.show()

# Ejemplo de predicción
numero_serial_nuevo = np.array([len("7702439001070") / np.max(X_data)], dtype=np.float32)
prediccion = model.predict(numero_serial_nuevo)
clase_predicha = np.argmax(prediccion)
print("Clase predicha:", list(clases.keys())[clase_predicha])
