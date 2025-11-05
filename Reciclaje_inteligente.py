# Script realizado por:
# Henrry Camilo Valencia - 2190564
# Juan Andrés Chacón - 2200015

# Imports
import cv2
import cv2.aruco as aruco
import numpy as np
from ultralytics import YOLO


# Cargando el modelo
best_model_weights = r"runs\detect\train8\weights\best.pt"
model = YOLO(best_model_weights)

# Obteniendo IP:
ip_address = 'http://192.168.1.3:8080/video' # Cambiar la IP donde se aloje su streaming de IP Webcam

cap = cv2.VideoCapture(0)
cap.open(ip_address)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

# De ahora en adelante será indispensable el módulo aruco de la librería cv2, ya que facilitará el uso de estos patrones
# Definir el diccionario ArUco y el tamaño del marcador en metros
# Se obtiene el diccionario de patrones aruco de cv2, en nuestro caso usaremos patrones 5x5 y solo usaremos 1 a la vez
# entonces usaremos el diccionario 5x5_50 que contiene 50 ids de 5x5 (diccionario más pequeño elegible)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)

# Se crea un objeto con todos los parámetros que le permiten a un detector identificar el Aruco
parameters = aruco.DetectorParameters()

# Creación del detector que usará los parámetros y el diccionario para identificar el Aruco cuando entre en escena
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Importante paso para la medición de los objetos, hace referencia al largo del lado del patrón Aruco en la vida real, para luego
# obtener la relación píxel/cm
marker_length = 0.095


# Función para calcular la escala de los bounding boxes
def get_scale(marker_corners, marker_length):
    # Se extraen las esquinas del marcador Aruco, se debe usar el detector con anterioridad
    corners = marker_corners[0][0]
    # Calcular la distancia euclidiana entre las esquinas del marcador
    dist = np.linalg.norm(corners[0] - corners[1])
    # Se obtiene la relación píxel/cm: (llamada "scale")
    scale = marker_length / dist
    return scale


# Ciclo principal para capturar imágenes de la cámara
while True:
    # Captura un frame de la cámara
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer el cuadro de la cámara")
        break

    # Detección del marcador ArUco, de aquí se obtienen los bordes para luego calcular la escala
    corners, ids, _ = detector.detectMarkers(frame)
    # Cuando se identifica el ID del Aruco, se puede medir con cierta presición el objeto por ello se genera un condicional el cual cuando detecta el Aruco en pantalla, aplica la medición de las clases.
    # (Cada patrón es diferente y tienen su propio identificador, en nuestro caso ID = 0)
    if ids is not None:
        # Calcular la escala
        scale = get_scale(corners, marker_length)

        # Realizar la detección en el cuadro capturado, este nivel de confiablididad fue también eurístico, de forma que se fue cambiando hasta obtener buen desempeño a diferentes distacias objeto - lente.
        results = model(frame, conf=0.65)

        # Obtener las cajas delimitadoras (bounding boxes) y las clasificaciones
        boxes = results[0].boxes
        classes = results[0].names

        # Iterar sobre cada bounding box para extraer las coordenadas, calcular dimensiones y obtener la clasificación
        for box in boxes:
            # Extraer las coordenadas (xmin, ymin, xmax, ymax)
            xmin, ymin, xmax, ymax = box.xyxy[
                0].cpu().numpy()  # Descubrimos que al procesarse los bb en la gpu, a la hora de extraerlos y aplicar numpy obteníamos un error, por lo tanto los datos de la gpu primero tienen que exportase hacia la cpu para aplicar numpy

            # Calcular el ancho y alto del bounding box
            width = xmax - xmin
            height = ymax - ymin

            # Obtener la etiqueta de cada clase
            class_id = int(box.cls[0].cpu().numpy())
            class_name = classes[class_id]

            # Calcular las dimensiones reales usando la escala, directamente lo transformamos a centímetros
            real_width = (width * scale) * 100
            real_height = (height * scale) * 100
            # Se realiza el cálculo del aproximado del volúmen, se asume un cilindro de radio constante.
            volume = np.pi * ((real_width // 2) ** 2) * real_height

            # Dibujar el bounding box en la imagen
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

            # Mostrar las dimensiones reales del bounding box y la etiqueta de clase en la imagen
            label = f'{class_name}: Ancho: {real_width:.2f}cm, Largo: {real_height:.2f}cm'

            cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Creamos otra etiqueta solo con la información del volumen, esto con el fin de imprimirla en la parte de debajo de la bb
            # y no saturar la parte superior
            label = f'Volumen: {volume:.2f}cm^3'

            # Coordenadas para colocar el texto en la parte inferior de la bounding box
            text_x = int(xmin)
            text_y = int(ymax) + 20  # Añade un desplazamiento para que el texto no esté pegado a la bounding box

            # Dibujar el texto en el frame
            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Dibujar el marcador ArUco detectado
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        # Mostrar el fotograma con las cajas delimitadoras y el marcador ArUco
        cv2.imshow("DETECCION", frame)

        # Debido a que cuando el patrón no está presente las medidas no son confiables, se eliminan estas de la ejecución mediante este
        # condicional (solo se genera el bb y la detección de clases)
    elif ids is None:
        # Realizar la detección en el cuadro capturado
        results = model(frame, conf=0.65)
        anotaciones = results[0].plot()
        cv2.imshow("DETECCION", anotaciones)

    # Código para cerrar la cámara con la tecla "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()