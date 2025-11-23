import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from ultralytics import YOLO
import time
import json
import joblib  # Para cargar tu modelo ML

# --- Configuración EAR y ojos ---
UMBRAL_EAR = 0.25
FRAMES_PARA_SOMNOLENCIA = 15
INDICES_OJO_IZQUIERDO = [362, 385, 387, 263, 373, 380]
INDICES_OJO_DERECHO = [33, 160, 158, 133, 153, 144]
CLASE_CELULAR_YOLO = 67

def calcular_ear(ojo):
    A = dist.euclidean(ojo[1], ojo[5])
    B = dist.euclidean(ojo[2], ojo[4])
    C = dist.euclidean(ojo[0], ojo[3])
    return (A + B) / (2.0 * C)

class FocusMonitorNN:
    def __init__(self, materia, modelo_ml_path):
        self.materia_actual = materia
        self.contador_frames_cerrados = 0
        self.registros_sesion = []
        self.frame_count = 0

        # Captura de contexto temporal
        now = time.localtime()
        self.fecha_inicio = time.strftime("%Y-%m-%d", now)
        self.hora_inicio = time.strftime("%H:%M:%S", now)
        self.dia_semana_inicio = time.strftime("%A", now)
        self.timestamp_inicio = time.time()

        # Modelos
        self.yolo_model = YOLO('yolov8n.pt')
        self.cap = cv2.VideoCapture(0)
        self.modelo_ml = joblib.load(modelo_ml_path)

    def iniciar_monitoreo(self):
        with mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        ) as face_mesh:

            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    continue

                celular_presente = False
                ear_promedio = 0.0
                enfoque_etiqueta = False

                # --- Detección celular ---
                if self.frame_count % 10 == 0:
                    yolo_results = self.yolo_model(frame, conf=0.4, verbose=False)[0]
                    for box in yolo_results.boxes:
                        if int(box.cls) == CLASE_CELULAR_YOLO:
                            celular_presente = True

                # --- Detección facial EAR ---
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resultados = face_mesh.process(rgb)

                if resultados.multi_face_landmarks:
                    landmarks = resultados.multi_face_landmarks[0]
                    shape = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) 
                             for lm in landmarks.landmark]

                    coords_izq = np.array([shape[i] for i in INDICES_OJO_IZQUIERDO])
                    coords_der = np.array([shape[i] for i in INDICES_OJO_DERECHO])
                    ear_promedio = (calcular_ear(coords_izq) + calcular_ear(coords_der)) / 2.0

                    if ear_promedio < UMBRAL_EAR:
                        self.contador_frames_cerrados += 1
                    else:
                        self.contador_frames_cerrados = 0

                    ojos_cerrados_sostenidos = self.contador_frames_cerrados >= FRAMES_PARA_SOMNOLENCIA
                    enfoque_etiqueta = not ojos_cerrados_sostenidos and not celular_presente

                # --- Guardar registro ---
                if resultados.multi_face_landmarks and self.frame_count % 10 == 0:
                    tiempo_actual = time.time() - self.timestamp_inicio
                    self.registros_sesion.append({
                        "timestamp": round(tiempo_actual, 2),
                        "celular_detectado": celular_presente,
                        "ear_promedio": round(ear_promedio, 4),
                        "enfoque_etiqueta": enfoque_etiqueta
                    })

                self.frame_count += 1
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()
        self.generar_json_final()

    def generar_json_final(self):
        if not self.registros_sesion:
            print("[ADVERTENCIA] No se encontraron registros")
            return

        registros = self.registros_sesion
        frames_totales = len(registros)
        frames_enfocados = sum(r['enfoque_etiqueta'] for r in registros)
        frames_distraidos = frames_totales - frames_enfocados
        porcentaje_enfoque = (frames_enfocados / frames_totales) * 100
        celular_frecuencia_relativa = sum(r['celular_detectado'] for r in registros) / frames_totales
        promedio_ear_general = np.mean([r['ear_promedio'] for r in registros])

        # --- Predicción de rendimiento con tu ML ---
        X = [[porcentaje_enfoque, celular_frecuencia_relativa, promedio_ear_general]]
        rendimiento = self.modelo_ml.predict(X)[0]  # bajo, neutro, alto

        json_sesion = {
            "materia": self.materia_actual,
            "contexto_temporal": {
                "fecha_inicio": self.fecha_inicio,
                "dia_semana": self.dia_semana_inicio,
                "hora_inicio": self.hora_inicio
            },
            "metricas_generales": {
                "porcentaje_enfocado": round(porcentaje_enfoque, 2),
                "frames_enfocados": frames_enfocados,
                "frames_distraidos": frames_distraidos,
                "promedio_ear_general": round(promedio_ear_general, 4),
                "celular_frecuencia_relativa": round(celular_frecuencia_relativa, 4)
            },
            "rendimiento_sesion": rendimiento,
            "datos_brutos_sesion": registros,
            "instruccion_gemini": "Analiza las métricas de enfoque y rendimiento para dar tips personalizados según si la sesión fue buena o necesita mejora."
        }

        nombre_archivo = f"analisis_{self.materia_actual}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(nombre_archivo, 'w') as f:
            json.dump(json_sesion, f, indent=4)

        print(f"[INFO] JSON generado: {nombre_archivo}")
        print(f"[INFO] Rendimiento de la sesión: {rendimiento}")

# --- Punto de entrada ---
if __name__ == "__main__":
    materia = input("Ingrese materia: ")
    monitor = FocusMonitorNN(materia, modelo_ml_path="modelo_incremental.pkl")
    monitor.iniciar_monitoreo()
