import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from ultralytics import YOLO 
import json
import time

# --- 1. Constantes y Funciones de Cálculo ---

UMBRAL_EAR = 0.25 
FRAMES_PARA_SOMNOLENCIA = 15 

INDICES_OJO_IZQUIERDO = [362, 385, 387, 263, 373, 380]
INDICES_OJO_DERECHO = [33, 160, 158, 133, 153, 144]
CLASE_CELULAR_YOLO = 67 

def calcular_ear(ojo):
    """Calcula la Relación de Aspecto del Ojo (EAR) basada en 6 puntos."""
    A = dist.euclidean(ojo[1], ojo[5])
    B = dist.euclidean(ojo[2], ojo[4])
    C = dist.euclidean(ojo[0], ojo[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- 2. Clase para encapsular la lógica ---

class FocusMonitor:
    def __init__(self, materia):
        # Atributos de la clase (reemplazan a las variables globales)
        self.materia_actual = materia
        self.contador_frames_cerrados = 0 
        self.registros_sesion = []
        self.timestamp_inicio = time.time()
        self.frame_count = 0
        
        print("[INFO] Cargando modelos...")
        self.yolo_model = YOLO('yolov8n.pt') 
        self.cap = cv2.VideoCapture(0)

    def iniciar_monitoreo(self):
        with mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Ignorando frame vacío.")
                    continue
                
                celular_presente = False
                ear_promedio = 0.0
                etiqueta_enfoque = False
                
                # --- A. Detección de Celular (YOLO) ---
                if self.frame_count % 10 == 0: 
                     yolo_results = self.yolo_model(frame, verbose=False)[0] 
                     for box in yolo_results.boxes:
                        if int(box.cls) == CLASE_CELULAR_YOLO:
                            celular_presente = True
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

                     coords_izq = np.array([shape[i] for i in INDICES_OJO_IZQUIERDO])
                     coords_der = np.array([shape[i] for i in INDICES_OJO_DERECHO])
                     ear_promedio = (calcular_ear(coords_izq) + calcular_ear(coords_der)) / 2.0
                     
                     # Lógica de Persistencia de Cierre de Ojos usando self.
                     if ear_promedio < UMBRAL_EAR:
                         self.contador_frames_cerrados += 1
                     else:
                         self.contador_frames_cerrados = 0
                     
                     ojos_cerrados_sostenidos = self.contador_frames_cerrados >= FRAMES_PARA_SOMNOLENCIA
                     etiqueta_enfoque = not ojos_cerrados_sostenidos and not celular_presente
                     
                     if etiqueta_enfoque:
                         estado_ojos = "ENFOCADO"
                         color_texto = (0, 255, 0) 
                     else:
                         if ojos_cerrados_sostenidos:
                             estado_ojos = "SOMNOLENCIA"
                         elif celular_presente:
                             estado_ojos = "CELULAR"
                         else: 
                             estado_ojos = "PARPADEANDO" 
                         color_texto = (0, 0, 255) 
                
                # --- C. Visualización y Almacenamiento EN MEMORIA ---
                cv2.putText(frame, f"EAR: {ear_promedio:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"FRAMES CERRADOS: {self.contador_frames_cerrados}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"CELULAR: {'SI' if celular_presente else 'NO'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"ESTADO: {estado_ojos}", (frame.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_texto, 2)

                # Registro de datos en la lista de memoria (cada 10 frames)
                if self.frame_count % 10 == 0 and resultados.multi_face_landmarks:
                    tiempo_actual = time.time() - self.timestamp_inicio
                    datos_registro = {
                        "timestamp": round(tiempo_actual, 2),
                        "celular_detectado": celular_presente,
                        "ear_promedio": round(ear_promedio, 4),
                        "enfoque_etiqueta": etiqueta_enfoque 
                    }
                    self.registros_sesion.append(datos_registro)
                    
                cv2.imshow("Detector de Enfoque Simplificado", frame)
                self.frame_count += 1
                
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

        # --- 4. Limpieza y EXPORTACIÓN A JSON ---
        self.cap.release()
        cv2.destroyAllWindows()
        self.exportar_json()


    def exportar_json(self):
        print("\n[INFO] Sesión finalizada. Analizando datos...")

        if self.registros_sesion:
            registros = self.registros_sesion
            frames_totales = len(registros)
            tiempos_enfoque = [r['enfoque_etiqueta'] for r in registros]
            frames_enfocados = sum(tiempos_enfoque)
            frames_distraidos = frames_totales - frames_enfocados
            
            porcentaje_enfoque = (frames_enfocados / frames_totales) * 100
            
            celular_en_distraccion = sum(r['celular_detectado'] for r in registros)
            celular_frecuencia_relativa = celular_en_distraccion / frames_distraidos if frames_distraidos > 0 else 0
            
            json_resumen = {
                "materia": self.materia_actual,
                "sesion_terminada_en": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "metricas_generales": {
                    "duracion_segundos": round(registros[-1]['timestamp'] - registros[0]['timestamp'], 2) if frames_totales > 0 else 0,
                    "total_registros": frames_totales,
                    "porcentaje_enfocado": round(porcentaje_enfoque, 2)
                },
                "patrones_distraccion": {
                    "celular_frecuencia_relativa": round(celular_frecuencia_relativa, 4), 
                    "promedio_ear_general": round(np.mean([r['ear_promedio'] for r in registros]), 4),
                },
                "datos_brutos_sesion": registros, 
                
                "instruccion_gemini": "Actúa como un coach de estudio. Analiza las métricas de enfoque y los datos brutos de la sesión. Proporciona un resumen de fortalezas/debilidades y 3 consejos de estudio accionables para mejorar el enfoque en la próxima sesión."
            }
            
            nombre_archivo_json = f"analisis_{self.materia_actual}_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(nombre_archivo_json, 'w') as f:
                json.dump(json_resumen, f, indent=4)
                
            print(f"[INFO] Análisis completo. JSON exportado como: {nombre_archivo_json}")

        else:
            print("[ADVERTENCIA] No se encontraron registros de enfoque para analizar. El JSON no fue creado.")


# --- 3. Punto de Entrada Principal ---

if __name__ == "__main__":
    try:
        materia_input = input("Por favor, ingrese la materia que estudiará (ej: Historia): ")
        monitor = FocusMonitor(materia_input)
        monitor.iniciar_monitoreo()
    except Exception as e:
        print(f"\n[ERROR CRÍTICO] El programa terminó inesperadamente: {e}")
