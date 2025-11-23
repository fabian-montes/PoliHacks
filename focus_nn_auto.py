import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from ultralytics import YOLO
import json
import time
import os
import joblib
from sklearn.linear_model import SGDClassifier

# ============================
# PARÁMETROS DE MONITOREO
# ============================
UMBRAL_EAR = 0.25
FRAMES_PARA_SOMNOLENCIA = 15
INDICES_OJO_IZQUIERDO = [362, 385, 387, 263, 373, 380]
INDICES_OJO_DERECHO = [33, 160, 158, 133, 153, 144]
CLASE_CELULAR_YOLO = 67

# ============================
# PARÁMETROS MODELO INCREMENTAL
# ============================
MODEL_FILE = "modelo_incremental_nn.pkl"
FEATURES_FILE = "dataset_incremental.jsonl"

# ============================
# FUNCIONES DE RED NEURONAL
# ============================
def crear_o_cargar_modelo(clases=None):
    if os.path.exists(MODEL_FILE):
        print(">> Modelo cargado.")
        return joblib.load(MODEL_FILE)
    
    print(">> Creando red neuronal incremental...")
    modelo = SGDClassifier(
        loss="log_loss",
        learning_rate="optimal",
        max_iter=1,
        warm_start=True
    )
    if clases is not None:
        modelo.partial_fit(np.zeros((1, len(clases))), [clases[0]], classes=clases)
    return modelo

def procesar_json(json_data):
    try:
        materia = json_data["materia"]
        hora = json_data["contexto_temporal"]["hora_inicio"]
        ear = json_data["patrones_distraccion"]["promedio_ear_general"]
        cel = json_data["patrones_distraccion"]["celular_frecuencia_relativa"]
        enfoque = json_data["metricas_generales"]["porcentaje_enfocado"]

        features = [
            hash(materia) % 999999,
            int(hora[:2]),
            float(ear),
            float(cel)
        ]

        # Clasificación simple: si porcentaje_enfocado > 70% → 1 (ENFOCADO), else 0
        label = 1 if enfoque >= 70 else 0
        return features, label
    except Exception as e:
        print("ERROR procesando JSON:", e)
        return None, None

def guardar_en_dataset(registro):
    with open(FEATURES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro) + "\n")

def cargar_dataset_completo():
    if not os.path.exists(FEATURES_FILE):
        return []
    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def entrenar_incremental(nombre_json):
    with open(nombre_json, "r", encoding="utf-8") as f:
        nuevo = json.load(f)

    print(f">> JSON '{nombre_json}' cargado para entrenamiento incremental.")
    guardar_en_dataset(nuevo)

    X_new, y_new = procesar_json(nuevo)
    if X_new is None:
        return
    
    dataset = cargar_dataset_completo()
    clases = sorted({1 if r["metricas_generales"]["porcentaje_enfocado"]>=70 else 0 for r in dataset})

    modelo = crear_o_cargar_modelo(clases=clases)
    modelo.partial_fit([X_new], [y_new], classes=clases)
    joblib.dump(modelo, MODEL_FILE)
    print(">> Modelo actualizado automáticamente con la nueva sesión.")

# ============================
# FUNCIONES DE MONITOREO
# ============================
def calcular_ear(ojo):
    A = dist.euclidean(ojo[1], ojo[5])
    B = dist.euclidean(ojo[2], ojo[4])
    C = dist.euclidean(ojo[0], ojo[3])
    return (A + B) / (2.0 * C)

class FocusMonitor:
    def __init__(self, materia):
        self.materia_actual = materia
        self.contador_frames_cerrados = 0
        self.registros_sesion = []
        now = time.localtime()
        self.fecha_inicio = time.strftime("%Y-%m-%d", now)
        self.hora_inicio = time.strftime("%H:%M:%S", now)
        self.dia_semana_inicio = time.strftime("%A", now)
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
                    continue
                
                celular_presente = False
                ear_promedio = 0.0
                etiqueta_enfoque = False

                # YOLO celular
                if self.frame_count % 10 == 0:
                    yolo_results = self.yolo_model(frame, conf=0.4, verbose=False)[0]
                    for box in yolo_results.boxes:
                        if int(box.cls) == CLASE_CELULAR_YOLO:
                            celular_presente = True
                            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                            cv2.putText(frame,"CELULAR",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resultados = face_mesh.process(rgb)
                estado_ojos = "BUSCANDO CARA..."
                color_texto = (0,0,255)

                if resultados.multi_face_landmarks:
                    landmarks = resultados.multi_face_landmarks[0]
                    shape = [(int(l.x*frame.shape[1]), int(l.y*frame.shape[0])) for l in landmarks.landmark]
                    coords_izq = np.array([shape[i] for i in INDICES_OJO_IZQUIERDO])
                    coords_der = np.array([shape[i] for i in INDICES_OJO_DERECHO])
                    ear_promedio = (calcular_ear(coords_izq) + calcular_ear(coords_der))/2.0

                    if ear_promedio < UMBRAL_EAR:
                        self.contador_frames_cerrados +=1
                    else:
                        self.contador_frames_cerrados =0

                    ojos_cerrados_sostenidos = self.contador_frames_cerrados>=FRAMES_PARA_SOMNOLENCIA
                    etiqueta_enfoque = not ojos_cerrados_sostenidos and not celular_presente

                    if etiqueta_enfoque:
                        estado_ojos="ENFOCADO"
                        color_texto=(0,255,0)
                    else:
                        if ojos_cerrados_sostenidos:
                            estado_ojos="SOMNOLENCIA"
                        elif celular_presente:
                            estado_ojos="CELULAR"
                        else:
                            estado_ojos="PARPADEANDO"
                        color_texto=(0,0,255)

                # Visualización
                cv2.putText(frame,f"EAR:{ear_promedio:.2f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                cv2.putText(frame,f"FRAMES CERRADOS:{self.contador_frames_cerrados}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                cv2.putText(frame,f"CELULAR:{'SI' if celular_presente else 'NO'}",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                cv2.putText(frame,f"ESTADO:{estado_ojos}",(frame.shape[1]-250,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,color_texto,2)

                if self.frame_count % 10 == 0 and resultados.multi_face_landmarks:
                    tiempo_actual = time.time()-self.timestamp_inicio
                    datos_registro = {
                        "timestamp": round(tiempo_actual,2),
                        "celular_detectado": celular_presente,
                        "ear_promedio": round(ear_promedio,4),
                        "enfoque_etiqueta": etiqueta_enfoque
                    }
                    self.registros_sesion.append(datos_registro)

                cv2.imshow("Detector de Enfoque Simplificado",frame)
                self.frame_count +=1

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

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
            porcentaje_enfoque = (frames_enfocados/frames_totales)*100
            celular_en_distraccion = sum(r['celular_detectado'] for r in registros)
            celular_frecuencia_relativa = celular_en_distraccion/frames_distraidos if frames_distraidos>0 else 0

            json_resumen={
                "materia":self.materia_actual,
                "contexto_temporal":{
                    "fecha_inicio":self.fecha_inicio,
                    "dia_semana":self.dia_semana_inicio,
                    "hora_inicio":self.hora_inicio
                },
                "sesion_terminada_en":time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()),
                "metricas_generales":{
                    "duracion_segundos":round(registros[-1]['timestamp']-registros[0]['timestamp'],2) if frames_totales>0 else 0,
                    "total_registros":frames_totales,
                    "porcentaje_enfocado":round(porcentaje_enfoque,2)
                },
                "patrones_distraccion":{
                    "celular_frecuencia_relativa":round(celular_frecuencia_relativa,4),
                    "promedio_ear_general":round(np.mean([r['ear_promedio'] for r in registros]),4)
                },
                "datos_brutos_sesion":registros,
                "instruccion_gemini":"Actúa como un coach de estudio. Analiza las métricas de enfoque, los datos brutos, la Materia y el Momento del día (hora y día de la semana) para dar consejos personalizados."
            }

            nombre_archivo_json=f"analisis_{self.materia_actual}_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(nombre_archivo_json,'w') as f:
                json.dump(json_resumen,f,indent=4)
            print(f"[INFO] JSON exportado como: {nombre_archivo_json}")

            # --- ENTRENAMIENTO AUTOMÁTICO INCREMENTAL ---
            entrenar_incremental(nombre_archivo_json)

        else:
            print("[ADVERTENCIA] No se encontraron registros. JSON no creado.")

# ============================
# EJECUCIÓN PRINCIPAL
# ============================
if __name__ == "__main__":
    materia_input = input("Por favor, ingrese la materia que estudiará: ")
    monitor = FocusMonitor(materia_input)
    monitor.iniciar_monitoreo()
