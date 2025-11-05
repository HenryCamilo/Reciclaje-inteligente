# Detección y medición de objetos con YOLOv8 y marcadores ArUco

**Autores:**
- Henry Camilo Valencia — 2190564
- Juan Andrés Chacón — 2200015

---

## Descripción general

Este proyecto implementa un **sistema de detección, clasificación y medición de objetos en tiempo real** mediante un **modelo YOLOv8** con pesos entrenados a medida y **marcadores ArUco** para la **calibración de escala**.

El objetivo principal es estimar las **dimensiones reales (en centímetros)** y el **volumen aproximado** de los objetos detectados a partir de una transmisión de vídeo en directo (p. ej., cámara web o cámara IP).

---

## Tecnologías y bibliotecas

- **Python 3.9+**

- **OpenCV (cv2)** — procesamiento de imágenes, detección ArUco y visualización.

- **NumPy** — operaciones matemáticas y manipulación de matrices.

- **Ultralytics YOLOv8** — modelo de detección de objetos.

- **AzureML (utilizado para el entrenamiento del modelo)** — gestión y experimentación del modelo.

- **ArUco (cv2.aruco)** — para la calibración de escala y la medición espacial.

---

## Objetivos del proyecto

1. **Detectar objetos en tiempo real** utilizando un modelo YOLOv8 entrenado.

2. **Reconocer marcadores ArUco** para calcular la relación píxel-centímetro (escala real).

3. **Medir las dimensiones reales (ancho y alto)** de los objetos detectados.

4. **Estimar el volumen aproximado** de cada objeto, asumiendo una forma cilíndrica.

5. **Mostrar los resultados visualmente** con recuadros delimitadores, etiquetas y medidas.

---

## Estructura del código

### 1. **Carga del modelo**
El script importa las bibliotecas necesarias y carga el modelo YOLOv8 entrenado.

```python
from ultralytics import YOLO
model = YOLO("runs/detect/train8/weights/best.pt")

```

---

### 2. **Configuración de la cámara**
El sistema admite:

- Una cámara web local (`cv2.VideoCapture(0)`), o

- Una transmisión de video IP (p. ej., la aplicación Android IP Webcam).

```python
ip_address = 'http://192.168.1.3:8080/video'
cap = cv2.VideoCapture(0)
cap.open(ip_address)

```

---

### 3. **Inicialización del marcador ArUco**

Se utiliza un diccionario **5x5_50** para una detección fiable en entornos controlados.

```python

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)

parameters = aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
marker_length = 0.095 # Longitud real del lado del marcador en metros

```

---

### 4. **Cálculo de escala (píxeles a cm)**
La función `get_scale()` calcula la relación píxeles/cm basándose en el tamaño real y el tamaño detectado del marcador.

```python
def get_scale(marker_corners, marker_length):

corners = marker_corners[0][0]

dist = np.linalg.norm(corners[0] - corners[1])

scale = marker_length / dist

return scale
```

---

### 5. **Bucle de detección principal**

- Captura fotogramas de vídeo.

- Detecta el marcador ArUco (si está visible).

- Ejecuta la detección de objetos YOLO.

- Calcula las dimensiones reales y el volumen.

```python
results = model(frame, conf=0.65)
boxes = results[0].boxes
```

---

### 6. **Medición y visualización**

Para cada objeto detectado:

- Dibuja un rectángulo delimitador.

- Calcula el **ancho**, la **altura** y el **volumen aproximado** (suponiendo forma cilíndrica). - Muestra las etiquetas de medición en el marco.

``python
ancho_real = (ancho * escala) * 100
alto_real = (alto * escala) * 100
volumen = np.pi * ((ancho_real // 2) ** 2) * alto_real

```

---

### 7. **Visualización de salida**
Muestra una ventana de detección en tiempo real titulada "DETECCIÓN".

Presione **Q** para salir.

---

## Fórmulas Matemáticas

1. **Escala (relación píxel/cm):**

\[ escala = \frac{longitud\_marcador}{distancia\_píxel} \]

2. **Dimensiones reales:**

\[ ancho\_real = (ancho \times escala) \times 100 \]

\[ alto\_real = (alto \times escala) \times 100 \]

3. **Volumen aproximado (cilindro):**

\[ V = \pi \times (\frac{ancho\_real}{2})^2 \times alto\_real \]

---

## Conceptos Clave

- **YOLOv8:** Modelo de detección de objetos en tiempo real desarrollado por Ultralytics.

- **Marcador ArUco:** Marcador fiducial cuadrado utilizado para la calibración y la estimación de la pose. **EDA (Análisis Exploratorio de Datos):** Paso previo al modelado para comprender la distribución y los patrones de los datos.

---

## Requisitos

Instalar las dependencias necesarias:

```bash

pip install opencv-python ultralytics numpy

```

---

## Ejecución del proyecto

1. Conectar la cámara o configurar la transmisión IP:

```python

ip_address = 'http://<TU_IP>:8080/video'

``` 2. Asegurarse de que existe el archivo del modelo entrenado (`best.pt`).

3. Ejecutar el script:

```bash

python detection_yolo_aruco.py

``` 4. Pulsar **Q** para detener el programa.

---

## Salida esperada

El sistema muestra:

- Detección de objetos en tiempo real.

- Cuadros delimitadores con:

- Nombre de la clase.

- Dimensiones reales (cm).

- Volumen estimado (cm³).

- Marcador ArUco utilizado para la calibración.

---
## Mejoras futuras

- Agregar varios marcadores ArUco para una mejor calibración.

- Implementar el registro de datos de las mediciones.

- Crear una interfaz gráfica o un panel web.

- Aplicar la calibración completa de la cámara para obtener resultados más precisos.

---
## Licencia
Este proyecto se desarrolló con fines académicos y de investigación como parte de una especialización.
Libre para usar y modificar, citando al autor.
