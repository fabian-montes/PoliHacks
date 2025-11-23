import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from ultralytics import YOLO 
import json
import time
import database 

# --- 1. Constantes y Funciones de Cálculo ---

# Umbral de EAR. (Si el valor cae por debajo de esto, se considera somnolencia)
UMBRAL_EAR = 0.25 
# *** NUEVAS CONSTANTES PARA EVITAR FALSOS POSITIVOS DE PARPADEO ***
FRAMES_PARA_SOMNOLENCIA = 15 # Ojos deben estar cerrados por 15 frames consecutivos (~0.5 segundos)

# NOTA: CONTADOR_FRAMES_CERRADOS se define FUERA del bucle, pero debe ser declarada como global dentro
# si se modifica.
CONTADOR_FRAMES_CERRADOS = 0 
# ********************************************************************

# Índices clave para el cálculo del EAR en MediaPipe Face Mesh
INDICES_OJO_IZQUIERDO = [362, 385, 387, 263, 373, 380]
INDICES_OJO_DERECHO = [33, 160, 158, 133, 153, 144]

# Mapeo del índice de YOLOv8 para "cell phone"
CLASE_CELULAR_YOLO = 67 

def calcular_ear(ojo):
    """Calcula la Relación de Aspecto del Ojo (EAR) basada en 6 puntos."""
    A = dist.euclidean(ojo[1], ojo[5])
    B = dist.euclidean(ojo[2], ojo[4])
    C = dist.euclidean(ojo[0], ojo[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- 2. Inicialización de Modelos y BD ---

print("[INFO] Cargando modelos...")
yolo_model = YOLO('yolov8n.pt') 

cap = cv2.VideoCapture(0)

timestamp_inicio = time.time()
frame_count = 0

# ********* 2.1 CONEXIÓN Y CONFIGURACIÓN DE LA BD *********
materia_actual = input("Por favor, ingrese la materia que estudiará (ej: Historia): ") 

conn = database.crear_conexion()
if not conn:
    print("No se pudo establecer la conexión con la BD. Saliendo...")
    exit() 
database.crear_tabla(conn) 
# **********************************************************

with mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:

    # --- 3. Bucle Principal de Video ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignorando frame vacío.")
            continue

        # *** DECLARACIÓN GLOBAL DENTRO DEL BUCLE ***
        # Necesario para modificar la variable global CONTADOR_FRAMES_CERRADOS
        #global CONTADOR_FRAMES_CERRADOS 
        # *******************************************
        
        celular_presente = False
        ear_promedio = 0.0
        
        # --- A. Detección de Celular (YOLO) ---
        if frame_count % 10 == 0: 
            yolo_results = yolo_model(frame, verbose=False)[0] 
            
            for box in yolo_results.boxes:
                if int(box.cls) == CLASE_CELULAR_YOLO:
                    celular_presente = True
                    # Dibujar cuadro delimitador (opcional)
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, 'CELULAR', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # --- B. Detección Facial y EAR (MediaPipe) ---
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = face_mesh.process(rgb)
        
        estado_ojos = "BUSCANDO CARA..."
        color_texto = (0, 0, 255)
        
        if resultados.multi_face_landmarks:
            landmarks = resultados.multi_face_landmarks[0]
            
            shape = []
            for id in range(len(landmarks.landmark)):
                x = int(landmarks.landmark[id].x * frame.shape[1])
                y = int(landmarks.landmark[id].y * frame.shape[0])
                shape.append((x, y))

            # Calcular EAR
            coords_izq = np.array([shape[i] for i in INDICES_OJO_IZQUIERDO])
            coords_der = np.array([shape[i] for i in INDICES_OJO_DERECHO])
            ear_promedio = (calcular_ear(coords_izq) + calcular_ear(coords_der)) / 2.0
            
            # *** Lógica de Persistencia de Cierre de Ojos ***
            if ear_promedio < UMBRAL_EAR:
                CONTADOR_FRAMES_CERRADOS += 1
            else:
                CONTADOR_FRAMES_CERRADOS = 0
            
            # Si el contador excede el umbral, se considera Somnolencia (ojos cerrados)
            ojos_cerrados_sostenidos = CONTADOR_FRAMES_CERRADOS >= FRAMES_PARA_SOMNOLENCIA
            
            # Etiqueta de Enfoque FINAL: Enfocado si NO hay somnolencia Y NO hay celular
            etiqueta_enfoque = not ojos_cerrados_sostenidos and not celular_presente
            # ******************************************************************
            
            if etiqueta_enfoque:
                estado_ojos = "ENFOCADO"
                color_texto = (0, 255, 0)  # Verde
            else:
                if ojos_cerrados_sostenidos:
                    estado_ojos = "SOMNOLENCIA"
                elif celular_presente:
                    estado_ojos = "CELULAR"
                else: 
                    estado_ojos = "PARPADEANDO" 
                
                color_texto = (0, 0, 255)  # Rojo
        
        # --- C. Visualización y Almacenamiento ---

        # Visualización
        cv2.putText(frame, f"EAR: {ear_promedio:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FRAMES CERRADOS: {CONTADOR_FRAMES_CERRADOS}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"CELULAR: {'SI' if celular_presente else 'NO'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"ESTADO: {estado_ojos}", (frame.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_texto, 2)

        # Almacenamiento de Datos (Cada 10 frames)
        if frame_count % 10 == 0 and resultados.multi_face_landmarks:
            tiempo_actual = time.time() - timestamp_inicio
            
            datos_registro = {
                "timestamp": round(tiempo_actual, 2),
                "celular_detectado": celular_presente,
                "ear_promedio": round(ear_promedio, 4),
                "enfoque_etiqueta": etiqueta_enfoque
            }
            
            # ********* INSERCIÓN EN LA BD *********
            database.insertar_registro(conn, datos_registro, materia_actual)
            # **************************************
        
        cv2.imshow("Detector de Enfoque Simplificado", frame)
        frame_count += 1
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# --- 4. Limpieza y Cierre ---
cap.release()
cv2.destroyAllWindows()

if conn:
    conn.close()
    print(f"\n[INFO] Recolección terminada. Datos guardados en {database.DB_FILE}")