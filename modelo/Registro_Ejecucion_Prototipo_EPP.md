# Registro de ejecución: prototipo de detección de EPP

**Proyecto:** Sistema de vigilancia inteligente para el uso correcto de EPP
**Equipo:** Johan Soto y Fernanda Alcaino
**Fecha de ejecución:** 20 de noviembre de 2026
**Código:** `modelo/codigo/` (instrucciones de uso en `modelo/README.md`)
**Evidencia:** `modelo/evidencia/`

Este registro documenta lo ejecutado en el prototipo y lo compara con lo planificado en el
*Plan de Proyecto EPP Consolidado v3*. El prototipo es una prueba de concepto de las fases
**F2 Ingeniería de datos**, **F3 Modelado** y **F4 Evaluación** de CRISP-ML(Q), hecha con un
dataset público mientras no existan imágenes propias de la planta.

---

## 1. Planificado frente a ejecutado

| Elemento | Planificado (Plan v3) | Ejecutado en el prototipo | Estado |
|---|---|---|---|
| Datos | Dataset propio de ≥ 10.000 imágenes anonimizadas de la planta (S3–S8) | Dataset público Construction-PPE: 1.000 imágenes de obras (800 entrenamiento, 200 validación), 14.863 cajas, 12 clases | Sustituto temporal |
| Exploración de datos | Revisión de calidad del etiquetado (compuerta F2) | Carga en FiftyOne, conteo por clase y mapa UMAP de embeddings MobileNet-v2 de cada caja, según la guía de Voxel51 | Ejecutado |
| Modelo | YOLOv8 con ajuste de hiperparámetros (S9–S12) | YOLO11n (sucesor de YOLOv8 en Ultralytics, misma API) con *fine-tuning* desde pesos COCO | Ejecutado, con cambio de versión |
| Entrenamiento | GPU del equipo de ML | 3 épocas en GPU RTX 3060 Ti y 10 épocas en CPU (la GPU falló, ver §4) | Ejecutado con desvío |
| Clases Must (RF-01) | Casco y chaleco | `helmet`, `vest`, `no_helmet`, `no_vest` (más guantes, botas, antiparras como Should) | Ejecutado |
| Alertas (RF-02) | Alerta si la omisión dura ≥ 3 s | Alerta inmediata en imágenes; en video se confirma tras 3 s continuos y no se repite por 30 s | Ejecutado |
| Latencia (RNF-01) | < 1 s por cuadro en el nodo Edge | 38 ms promedio y 56 ms máximo por imagen en CPU de escritorio | Cumple en PC; falta medir en el nodo Edge |
| Desempeño (RNF-02) | Recall ≥ 0,95 y precisión ≥ 0,90 por clase | Precisión 0,91 y recall 0,70 globales; recall de `no_helmet` 0,50 y `no_vest` 0,58 | No cumple aún (esperado) |
| Privacidad (RF-03, exclusiones) | Difuminado de rostros, sin identificación | Sin identificación de personas; el registro guarda solo fecha, origen, tipo y confianza. El difuminado no está implementado | Parcial |
| Validación humana (RF-04) | Prevencionista valida cada alerta | `alertas.csv` y la App de FiftyOne permiten la revisión, pero sin flujo formal | Pendiente |

## 2. Pasos ejecutados

| # | Paso | Script | Duración aprox. | Resultado |
|---|---|---|---|---|
| 1 | Entorno: Python 3.11 con uv, PyTorch 2.6 (CUDA 12.4), Ultralytics 8.4.216, FiftyOne 1.22.1 | — | 10 min | `.venv` funcional |
| 2 | Descarga del dataset (650 MB) y YAML | `01_descargar_datos.py` | 3 min | 1.000 imágenes; 12 copiadas a `modelo/images/` |
| 3 | Exploración en FiftyOne con embeddings y UMAP | `02_explorar_fiftyone.py` | 30 min en CPU | Dataset persistente `epp-construction-ppe` |
| 4 | Entrenamiento YOLO11n | `03_entrenar_yolo.py` | 2 min en GPU + 40 min en CPU | `modelo/pesos/epp_yolo11n.pt` (5,5 MB) |
| 5 | Evaluación en FiftyOne | `04_evaluar_fiftyone.py` | 1 min | `evaluacion.txt` |
| 6 | Demo de alertas sobre 12 imágenes | `05_detectar_alertas.py` | < 1 min | 5 imágenes con alerta, `alertas.csv` |

## 3. Resultados

### 3.1 Distribución de clases (FiftyOne)

| Clase | Cajas | Clase | Cajas |
|---|---|---|---|
| boots | 2.484 | no_gloves | 1.042 |
| Person | 2.443 | no_helmet | 813 |
| gloves | 1.929 | goggles | 751 |
| vest | 1.816 | no_boots | 747 |
| helmet | 1.576 | no_vest | 567 |
| forklift | 348 | no_goggle | 347 |

Las clases de incumplimiento (`no_*`) tienen entre 2 y 3 veces menos ejemplos que las de
cumplimiento. Este desbalance explica gran parte del bajo recall en esas clases.

### 3.2 Evolución del entrenamiento (validación, 200 imágenes)

| Etapa | Época | mAP50 | mAP50-95 |
|---|---|---|---|
| GPU | 1 | 0,406 | 0,259 |
| GPU | 3 | 0,597 | 0,392 |
| CPU (parte del mejor peso GPU) | 1 | 0,666 | 0,453 |
| CPU | 4 | 0,687 | 0,464 |
| CPU | 8 | 0,726 | 0,504 |
| **Modelo final (CPU)** | 10 | **0,727** | **0,513** |

### 3.3 Métricas por clase (FiftyOne, IoU 0,5, confianza ≥ 0,35)

| Clase | Precisión | Recall | Clase | Precisión | Recall |
|---|---|---|---|---|---|
| helmet | 0,92 | 0,76 | no_helmet | 0,86 | **0,50** |
| vest | 0,94 | 0,85 | no_vest | 0,97 | **0,58** |
| gloves | 0,92 | 0,72 | no_gloves | 0,88 | 0,51 |
| boots | 0,87 | 0,63 | no_boots | 0,95 | 0,61 |
| goggles | 0,92 | 0,94 | no_goggle | 0,64 | 0,11 |
| Person | 0,95 | 0,86 | forklift | 0,80 | 0,77 |

Total: 2.075 verdaderos positivos, 197 falsos positivos y 886 falsos negativos.
mAP COCO (0,50–0,95) calculado por FiftyOne: 0,471.

**Lectura:** el modelo casi no da falsas alarmas (precisión 0,91), pero deja pasar cerca de la
mitad de las omisiones de casco y chaleco. Para seguridad esto es lo más grave, porque una
omisión no detectada no genera alerta. Es coherente con un prototipo de 1.000 imágenes y
10 épocas, y confirma que la meta del plan (recall ≥ 0,95) requiere el dataset propio y el
entrenamiento completo de la fase F3.

## 4. Desviaciones y problemas

1. **Dataset distinto al de la guía.** La guía de Voxel51 usa SH17 (8.099 imágenes, 14 GB,
   requiere cuenta de Kaggle). Se usó Construction-PPE por ser 20 veces más liviano y por tener
   clases explícitas de ausencia de EPP. El código permite cambiar a SH17 (ver README §4).
2. **YOLO11 en vez de YOLOv8.** Ultralytics recomienda YOLO11 como versión actual; la forma de
   entrenar y usar es idéntica. El plan puede actualizarse o mantenerse, porque cambiar a
   `yolov8n.pt` es un solo argumento (`--modelo yolov8n.pt`).
3. **Falla de la GPU.** La RTX 3060 Ti se desconectó dos veces durante la validación de una época
   ("GPU is lost", requiere reiniciar Windows). Se guardaron los mejores pesos de la época 3 y
   se continuó en CPU. Riesgo para el plan: el hardware de entrenamiento puede fallar; mitigación
   propuesta: guardar pesos por época (Ultralytics ya lo hace) y tener un plan B en la nube.
4. **Imágenes de la demo.** Las 12 imágenes de `modelo/images/` vienen del conjunto de validación, que
   el modelo no vio al entrenar, pero son del mismo origen. Falta probar con imágenes de una
   planta real.

## 5. Próximos pasos

1. Reiniciar el equipo y reentrenar en GPU con más épocas (`--epocas 100`) y YOLO11s, para medir
   cuánto sube el recall de `no_helmet` y `no_vest`.
2. Bajar el umbral de confianza para las clases de alerta y medir el cambio de recall y
   falsas alarmas (curva precisión–recall).
3. Agregar el difuminado de rostros (RF-03) antes de guardar cualquier imagen anotada.
4. Probar la demo con un video real (`--fuente video.mp4`) para validar la regla de 3 segundos.
5. Medir la latencia en el hardware Edge definido en la arquitectura.

## 6. Evidencia (`modelo/evidencia/`)

| Archivo | Contenido |
|---|---|
| `demo_alerta_1.jpg`, `demo_alerta_2.jpg` | Imágenes con alerta: sin casco, sin chaleco, sin guantes |
| `demo_ok.jpg` | Imagen sin incumplimientos |
| `alertas.csv` | Registro de alertas generado por la demo |
| `evaluacion.txt` | Informe de FiftyOne por clase |
| `curvas_entrenamiento.png` | Pérdidas y mAP por época (etapa CPU) |
| `matriz_confusion.png` | Matriz de confusión normalizada |
| `fiftyone_embeddings.jpg` | Mapa UMAP de embeddings por clase en la App de FiftyOne |
| `resultados_gpu_epocas1-3.csv`, `resultados_cpu_epocas1-10.csv` | Métricas por época |

Licencia del dataset: Construction-PPE, Ultralytics, CC BY 4.0
(<https://docs.ultralytics.com/datasets/detect/construction-ppe>).
