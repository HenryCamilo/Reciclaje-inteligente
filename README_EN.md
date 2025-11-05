# 🧠 Object Detection and Measurement with YOLOv8 and ArUco Markers

**Authors:**  
- 👨‍💻 Henry Camilo Valencia — 2190564  
- 👨‍💻 Juan Andrés Chacón — 2200015  

---

## 📖 Overview

This project implements a **real-time object detection, classification, and measurement system** using a **YOLOv8 model** with custom-trained weights and **ArUco markers** for **scale calibration**.

The main goal is to estimate the **real-world dimensions (in centimeters)** and **approximate volume** of detected objects from a live video stream (e.g., webcam or IP camera).  

---

## ⚙️ Technologies and Libraries

- **Python 3.9+**
- **OpenCV (cv2)** — image processing, ArUco detection, and visualization.
- **NumPy** — mathematical operations and matrix handling.
- **Ultralytics YOLOv8** — object detection model.
- **AzureML (used for model training)** — model management and experimentation.
- **ArUco (cv2.aruco)** — for scale calibration and spatial measurement.

---

## 🎯 Project Objectives

1. **Detect objects in real time** using a YOLOv8 trained model.  
2. **Recognize ArUco markers** to calculate the pixel-to-centimeter ratio (real-world scale).  
3. **Measure real dimensions (width and height)** of detected objects.  
4. **Estimate the approximate volume** of each object, assuming a cylindrical shape.  
5. **Display results visually** with bounding boxes, labels, and measurements.  

---

## 🧩 Code Structure

### 1. **Model Loading**
The script imports required libraries and loads the trained YOLOv8 model.

```python
from ultralytics import YOLO
model = YOLO("runs/detect/train8/weights/best.pt")
```

---

### 2. **Camera Configuration**
The system supports either:  
- A local webcam (`cv2.VideoCapture(0)`), or  
- An IP video stream (e.g., Android IP Webcam app).

```python
ip_address = 'http://192.168.1.3:8080/video'
cap = cv2.VideoCapture(0)
cap.open(ip_address)
```

---

### 3. **ArUco Marker Initialization**
A **5x5_50** dictionary is used for reliable detection in controlled environments.

```python
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
parameters = aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
marker_length = 0.095  # Real marker side length in meters
```

---

### 4. **Scale Calculation (pixels to cm)**
The `get_scale()` function calculates the pixel/cm ratio based on the marker's real and detected size.

```python
def get_scale(marker_corners, marker_length):
    corners = marker_corners[0][0]
    dist = np.linalg.norm(corners[0] - corners[1])
    scale = marker_length / dist
    return scale
```

---

### 5. **Main Detection Loop**
- Captures video frames.  
- Detects the ArUco marker (if visible).  
- Runs YOLO object detection.  
- Calculates real dimensions and volume.

```python
results = model(frame, conf=0.65)
boxes = results[0].boxes
```

---

### 6. **Measurement and Visualization**
For each detected object:
- Draws a bounding box.  
- Calculates **width**, **height**, and **approximate volume** (cylindrical assumption).  
- Displays measurement labels on the frame.

```python
real_width = (width * scale) * 100
real_height = (height * scale) * 100
volume = np.pi * ((real_width // 2) ** 2) * real_height
```

---

### 7. **Output Display**
Displays a real-time detection window titled `"DETECCION"`.  
Press **Q** to exit.  

---

## 📏 Mathematical Formulas

1. **Scale (pixel/cm ratio):**  
   \[ scale = \frac{marker\_length}{pixel\_distance} \]

2. **Real-world dimensions:**  
   \[ real\_width = (width \times scale) \times 100 \]  
   \[ real\_height = (height \times scale) \times 100 \]

3. **Approximate volume (cylinder):**  
   \[ V = \pi \times (\frac{real\_width}{2})^2 \times real\_height \]

---

## 🧠 Key Concepts

- **YOLOv8:** Real-time object detection model developed by Ultralytics.  
- **ArUco Marker:** Square fiducial marker used for calibration and pose estimation.  
- **EDA (Exploratory Data Analysis):** Pre-modeling step to understand data distribution and patterns.  

---

## 🧰 Requirements

Install required dependencies:

```bash
pip install opencv-python ultralytics numpy
```

---

## 🚀 Running the Project

1. Connect your camera or configure the IP stream:  
   ```python
   ip_address = 'http://<YOUR_IP>:8080/video'
   ```
2. Ensure that the trained model file (`best.pt`) exists.  
3. Run the script:  
   ```bash
   python detection_yolo_aruco.py
   ```
4. Press **Q** to stop the program.  

---

## 📸 Expected Output

The system displays:  
- Real-time object detection.  
- Bounding boxes with:  
  - Class name.  
  - Real dimensions (cm).  
  - Estimated volume (cm³).  
- ArUco marker used for calibration.  

---

## 📚 Future Improvements

- Add multiple ArUco markers for better calibration.  
- Implement data logging of measurements.  
- Create a graphical interface or web dashboard.  
- Apply full camera calibration for more accurate results.  

---

## 🧾 License

This project was developed for **academic and research purposes** as part of a specialization in **Artificial Intelligence**.  
Free to use and modify with author attribution.
