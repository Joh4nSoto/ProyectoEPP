# Modelo de detección de EPP: guía técnica

Documentación técnica del prototipo: datos, entrenamiento, evaluación, inferencia y alertas.
La visión de negocio y de gestión del proyecto está en el [README principal](../README.md).

El prototipo detecta EPP en imágenes, video o cámara web y genera una **alerta** cuando un
trabajador no usa casco, chaleco, guantes, botas o antiparras. Corre 100 % local, sin internet
después de la instalación (RNF-03). Sigue el flujo de la guía de Voxel51
[PPE Detection and Safety Monitoring](https://docs.voxel51.com/getting_started/manufacturing/10_ppe_detection.html)
(dataset YOLO en FiftyOne y embeddings) y agrega el entrenamiento y la evaluación de un detector
YOLO de Ultralytics.

## Ficha técnica

| Aspecto | Valor |
|---|---|
| Tarea | Detección de objetos (cajas + clase + confianza) |
| Arquitectura | YOLO11n (Ultralytics), 2,6 M de parámetros, pesos iniciales COCO |
| Pesos entrenados | `pesos/epp_yolo11n.pt` (5,5 MB) |
| Dataset | Construction-PPE (Ultralytics, CC BY 4.0): 800 de entrenamiento y 200 de validación, 14.863 cajas, 12 clases |
| Entrada | Imagen RGB redimensionada a 640 × 640 |
| Entrenamiento | 13 épocas: 3 en GPU (RTX 3060 Ti) + 10 en CPU desde el mejor peso; batch 16, semilla 42, aumentación por defecto de Ultralytics |
| Umbral de confianza | 0,35 (`CONF_MINIMA` en `codigo/config.py`) |
| Métricas (validación) | mAP50 0,727 · mAP50-95 0,513 · precisión 0,91 · recall 0,70 |
| Recall de clases de alerta | no_helmet 0,50 · no_vest 0,58 · no_gloves 0,51 · no_boots 0,61 · no_goggle 0,11 |
| Latencia | 38 ms promedio, 56 ms máximo por imagen (CPU de escritorio, pre + inferencia + post) |
| Stack | Python 3.11.15, PyTorch 2.6.0 (CUDA 12.4), Ultralytics 8.4.216, FiftyOne 1.22.1, OpenCV, UMAP |

## Pipeline

```
01_descargar_datos ──► 02_explorar_fiftyone ──► 03_entrenar_yolo ──► 04_evaluar_fiftyone
   dataset + YAML        embeddings + UMAP        pesos/epp_yolo11n.pt    métricas por clase
                                                          │
                                                          ▼
                       cámara / video / imagen ──► 05_detectar_alertas ──► alertas.csv + imágenes anotadas
                                                   (DetectorEPP en modelo.py: detectar → reglas → dibujar)
```

---

## 1. Estructura

Todo lo del modelo está en esta carpeta, `modelo/`. El entorno de Python (`.venv`) queda en la
raíz del proyecto, y los comandos se ejecutan desde ahí (`E:\ProyectoEPP`).

```
modelo/
├── README.md                            esta guía
├── Registro_Ejecucion_Prototipo_EPP.md  planificado frente a ejecutado (EP5)
├── codigo/
│   ├── config.py                 rutas, clases y reglas de alerta
│   ├── 01_descargar_datos.py     descarga el dataset y prepara images/
│   ├── 02_explorar_fiftyone.py   carga en FiftyOne + embeddings (guía Voxel51)
│   ├── 03_entrenar_yolo.py       fine-tuning de YOLO11n
│   ├── 04_evaluar_fiftyone.py    métricas por clase con FiftyOne
│   ├── 05_detectar_alertas.py    demo: detección + alertas
│   ├── modelo.py                 clase DetectorEPP (detección, reglas, dibujo)
│   └── requirements.txt
├── pesos/        epp_yolo11n.pt entrenado (se versiona) y yolo11n.pt base (se descarga solo)
├── images/       imágenes de prueba para la demo
├── evidencia/    capturas, métricas y alertas del prototipo
├── datos/        dataset descargado y su YAML (no se versiona)
├── resultados/   salidas de la demo (no se versiona)
└── runs/         gráficos y registros de entrenamiento de Ultralytics (no se versiona)
```

## 2. Instalación (Windows, una sola vez)

### 2.1 Demo con doble clic (recomendado)

En la raíz del repositorio está **`Iniciar_Demo_EPP.bat`**. Basta con clonar el repositorio y
hacer doble clic en ese archivo para abrir la demo en tiempo real con la cámara web:

1. Si el entorno `.venv` no existe o está incompleto, lo prepara solo: instala
   [uv](https://docs.astral.sh/uv/) si falta (necesita internet), Python 3.11 dentro de
   `.python\`, PyTorch (con CUDA 12.4 si detecta una GPU NVIDIA, si no la versión CPU) y
   `modelo\codigo\requirements.txt`. La primera vez tarda entre 5 y 15 minutos según la conexión;
   las siguientes parte de inmediato.
2. Deja OpenCV con soporte de ventanas (ver sección 9).
3. Lanza `05_detectar_alertas.py --fuente 0` con los pesos versionados `modelo\pesos\epp_yolo11n.pt`.

En la ventana de video, **q** cierra la demo. Las alertas quedan en `modelo\resultados\alertas.csv`.
Si el equipo tiene más de una cámara, se puede elegir otra desde una terminal:

```powershell
.\Iniciar_Demo_EPP.bat 1
```

Todas las rutas son relativas a la carpeta del `.bat`, así que funciona desde cualquier unidad o carpeta.

### 2.2 Instalación manual

Se usa [uv](https://docs.astral.sh/uv/) para crear el entorno con Python 3.11.

```powershell
cd E:\ProyectoEPP
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"   # Python dentro del proyecto, no en AppData
uv python install 3.11
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv\Scripts\python.exe -r modelo\codigo\requirements.txt
uv pip uninstall --python .venv\Scripts\python.exe opencv-python-headless   # ver sección 9
uv pip install --python .venv\Scripts\python.exe --no-deps --reinstall-package opencv-python -r modelo\codigo\requirements.txt
```

Sin GPU NVIDIA, se puede omitir la línea de PyTorch con `cu124`: todo funciona en CPU, solo más lento.
No hace falta activar el entorno: basta con llamar a `.venv\Scripts\python.exe`.

## 3. Uso paso a paso

Desde `E:\ProyectoEPP`, con `$py = ".venv\Scripts\python.exe"`:

| Paso | Comando | Resultado |
|---|---|---|
| 1. Datos | `& $py modelo\codigo\01_descargar_datos.py` | `modelo\datos\construction-ppe` (1.000 imágenes) y 12 imágenes en `modelo\images\` |
| 2. Explorar | `& $py modelo\codigo\02_explorar_fiftyone.py` | App de FiftyOne en http://localhost:5151 con el mapa de embeddings |
| 3. Entrenar | `& $py modelo\codigo\03_entrenar_yolo.py --epocas 30` | `modelo\pesos\epp_yolo11n.pt` y gráficos en `modelo\runs\entrenamiento_epp` |
| 4. Evaluar | `& $py modelo\codigo\04_evaluar_fiftyone.py --app` | `modelo\resultados\evaluacion.txt` con precisión, recall y mAP por clase |
| 5. Demo | `& $py modelo\codigo\05_detectar_alertas.py` | `modelo\resultados\imagenes\` anotadas y `modelo\resultados\alertas.csv` |

Variantes de la demo:

```powershell
.venv\Scripts\python.exe modelo\codigo\05_detectar_alertas.py --fuente C:\ruta\foto.jpg    # una imagen
.venv\Scripts\python.exe modelo\codigo\05_detectar_alertas.py --fuente C:\ruta\video.mp4   # un video (genera _anotado.mp4)
.venv\Scripts\python.exe modelo\codigo\05_detectar_alertas.py --fuente 0                   # cámara web, q para salir
.venv\Scripts\python.exe modelo\codigo\05_detectar_alertas.py --conf 0.5                   # más exigente con la confianza
```

En video, una alerta se confirma solo si la omisión dura **3 segundos** seguidos (requisito del
plan) y luego no se repite por 30 s, para no saturar al prevencionista.

## 4. Datos utilizados

La guía de Voxel51 usa el dataset **SH17** (8.099 imágenes, 17 clases, 14 GB en Kaggle,
licencia CC BY-NC-SA 4.0). Para tener una demo rápida se eligió **Construction-PPE** de
Ultralytics (1.000 imágenes, 650 MB, licencia CC BY 4.0), por dos razones:

1. Es 20 veces más liviano y se descarga sin cuenta de Kaggle.
2. Incluye clases de **ausencia** de EPP (`no_helmet`, `no_vest`, `no_gloves`, `no_boots`,
   `no_goggle`). Así el modelo detecta directamente el incumplimiento y la alerta es una regla
   simple, sin tener que cruzar geométricamente cajas de persona y de casco.

Clases: `helmet`, `gloves`, `vest`, `boots`, `goggles`, `Person`, `no_helmet`, `no_goggle`,
`no_gloves`, `no_boots`, `forklift`, `no_vest`. Separación: 800 imágenes de entrenamiento y
200 de validación.

Para escalar al dataset SH17 de la guía: descargar desde
<https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection>, crear un YAML con
sus 17 clases y apuntar `DATASET_YAML` en `modelo/codigo/config.py`. Las reglas de alerta tendrían que
cambiar, porque SH17 no tiene clases `no_*` (habría que cruzar `Head` sin `Helmet`).

## 5. Cómo se usa YOLO en este proyecto

**YOLO** (*You Only Look Once*) es una familia de redes neuronales de detección de objetos que,
en una sola pasada sobre la imagen, entrega cajas (x1, y1, x2, y2), una clase y una confianza para
cada objeto. Se usa la implementación de **Ultralytics** (`pip install ultralytics`).

### 5.1 Transfer learning

No se entrena desde cero. Se parte de `yolo11n.pt`, la versión *nano* de YOLO11 (2,6 millones de
parámetros) ya entrenada con el dataset COCO (80 clases generales). Esa red ya sabe reconocer
bordes, formas y personas; el *fine-tuning* solo le enseña las 12 clases de EPP:

```python
from ultralytics import YOLO

modelo = YOLO("yolo11n.pt")                    # pesos preentrenados en COCO
modelo.train(data="datos/construction-ppe.yaml",  # rutas y nombres de clases
             epochs=30, imgsz=640, batch=16, device=0)
```

- `data`: YAML con la carpeta del dataset y la lista de clases. Las etiquetas están en formato
  YOLO: un `.txt` por imagen con líneas `clase x_centro y_centro ancho alto` normalizadas a 0–1.
- `epochs`: pasadas completas por el set de entrenamiento.
- `imgsz`: las imágenes se redimensionan a 640 px.
- `device`: `0` para la GPU, `"cpu"` para el procesador.

Durante el entrenamiento, Ultralytics aplica aumentación de datos (mosaico, cambios de color,
volteos), valida en cada época y guarda el mejor modelo (`best.pt`) según el mAP.

### 5.2 Métricas

- **Precisión:** de las cajas que predijo el modelo, cuántas eran correctas.
- **Recall (exhaustividad):** de los objetos reales, cuántos encontró. Es la métrica clave para
  seguridad, porque un `no_helmet` no detectado es un riesgo no alertado. El plan fija una meta
  de recall ≥ 0,95 para producción.
- **mAP50:** precisión media cuando una caja cuenta como correcta si se superpone al menos un 50 %
  con la real. **mAP50-95** promedia umbrales de 50 % a 95 % (más estricto).

### 5.3 Inferencia y alertas

```python
resultado = modelo.predict("images/foto.jpg", conf=0.35)[0]
for caja, clase, conf in zip(resultado.boxes.xyxy, resultado.boxes.cls, resultado.boxes.conf):
    print(resultado.names[int(clase)], float(conf), caja.tolist())
```

`modelo.py` envuelve esto en la clase `DetectorEPP`. Cada detección de una clase `no_*` se
convierte en una alerta con severidad (casco y chaleco ALTA, el resto MEDIA, según la
priorización MoSCoW del plan). Las reglas están en `config.py` (`INCUMPLIMIENTOS`), así que se
pueden ajustar sin tocar el código.

### 5.4 Papel de FiftyOne

FiftyOne no entrena el modelo: sirve para **entender los datos y los errores**.

- `02_explorar_fiftyone.py` importa las etiquetas YOLO, calcula un embedding (vector de
  características) de cada caja con MobileNet-v2 y los proyecta a 2D con UMAP. En la App se ven
  grupos por clase; si `no_helmet` y `helmet` se mezclan, el modelo los confundirá.
- `04_evaluar_fiftyone.py` agrega las predicciones de YOLO al dataset, marca cada caja como
  verdadero positivo, falso positivo o falso negativo, y ordena las imágenes por errores para
  revisarlas una a una.

## 6. Configuración

Todo lo ajustable está en `codigo/config.py`:

| Variable | Uso |
|---|---|
| `CLASES` | Nombres de las 12 clases, en el orden de los índices del dataset |
| `INCUMPLIMIENTOS` | Clase → (severidad, mensaje). Casco y chaleco son ALTA (Must); guantes, botas y antiparras, MEDIA (Should) |
| `CONF_MINIMA` | Confianza mínima de una detección (0,35). Bajarla sube el recall y también las falsas alarmas |
| `MODELO_BASE` / `MODELO_ENTRENADO` | Pesos de partida y pesos finales |
| `DATASET_YAML` | YAML del dataset; cambiarlo permite entrenar con otro dataset (por ejemplo, SH17) |

Las reglas de tiempo del video están en `05_detectar_alertas.py`: la alerta se confirma tras
`SEGUNDOS_PERSISTENCIA = 3` segundos continuos y se silencia 30 s.

Formato de `alertas.csv`: `fecha, origen, severidad, mensaje, clase, confianza`.

## 7. Resultados y análisis de errores

El detalle por clase, la evolución por época y la evidencia están en
[`Registro_Ejecucion_Prototipo_EPP.md`](Registro_Ejecucion_Prototipo_EPP.md) y en `evidencia/`
(curvas de entrenamiento, matriz de confusión, mapa de embeddings y ejemplos de alerta).

- **Desbalance:** las clases `no_*` tienen de 2 a 3 veces menos cajas que las de cumplimiento.
  Es la causa principal del bajo recall en las alertas.
- **Precisión alta, recall bajo:** con 2.075 VP, 197 FP y 886 FN, el modelo es conservador:
  cuando alerta suele acertar, pero deja pasar omisiones.
- **`no_goggle` (recall 0,11):** es la clase más pequeña (347 cajas) y la de objeto más chico.
  No es una clase Must en el plan.

Mejoras técnicas para la fase F3: más épocas (100+) con GPU, YOLO11s o superior, `imgsz=960` para
objetos pequeños, balanceo de clases (sobremuestreo de imágenes con `no_*`), umbral de confianza
por clase y el dataset propio de la planta, con al menos un 20 % de imágenes nocturnas.

## 8. Sesgos, ética y límites del modelo

El análisis ético completo del proyecto está en el
[Anexo C](../entregables/Plan/Anexo_C_Evaluacion_Etica_y_Legal.pdf). Esta sección cubre los
sesgos propios del prototipo y cómo se controlan en el código.

### Sesgos conocidos del prototipo

| Sesgo | Origen | Efecto | Mitigación |
|---|---|---|---|
| **Dominio** | Las 1.000 imágenes son de obras de construcción y bodegas, no de la planta del cliente | El rendimiento en la planta real es desconocido y probablemente menor | Reentrenar con el dataset propio (E4) antes de cualquier uso real |
| **Iluminación** | Casi todas las imágenes son diurnas y con buena luz | Más falsos negativos en zonas oscuras y en el turno de noche, siempre contra los mismos trabajadores | ≥ 20 % de imágenes nocturnas en E4 y evaluar el recall por condición de luz (campo de metadatos en FiftyOne) |
| **Desbalance de clases** | Las clases `no_*` tienen de 2 a 3 veces menos cajas | Recall bajo justo en las clases que generan alertas | Sobremuestreo, más datos de incumplimiento y umbral de confianza por clase |
| **Demográfico** | El dataset no registra complexión, género ni tono de piel, así que no se puede medir | Posible detección desigual de personas o de su EPP | En el dataset propio, etiquetar atributos no identificatorios de forma agregada y comparar el recall por grupo antes del piloto |
| **Tamaño del objeto** | Antiparras y guantes ocupan pocos píxeles a 640 px | `no_goggle` con recall de 0,11 | `imgsz` mayor o cámaras más cercanas en zonas con antiparras |

### Controles de diseño en el código

- **Sin identificación:** el modelo no tiene clases de rostro ni de identidad, y `alertas.csv`
  guarda solo fecha, origen, severidad, clase y confianza.
- **Sin sanción automática:** el código solo registra y muestra alertas. La decisión siempre es
  de un prevencionista (RF-04), que puede revisar los errores en la App de FiftyOne.
- **Menos falsas alarmas:** en video, la alerta exige 3 segundos continuos de omisión, lo que evita
  alertas por ocultamientos breves.
- **Umbral explícito:** `CONF_MINIMA` está en `config.py` y se documenta su efecto sobre recall y
  falsas alarmas, para que el ajuste sea una decisión consciente.

### Límites actuales

- El difuminado facial (RF-03) aún no está implementado. Las imágenes anotadas de
  `resultados/` muestran rostros, así que no deben compartirse fuera del equipo. La imagen de la
  presentación (`demo_alerta_1_blur.jpg`) se difuminó por separado.
- El recall de casco y chaleco (0,50 y 0,58) no permite usar el prototipo como sistema de
  seguridad. Es solo una prueba de concepto.

## 9. Problemas frecuentes

| Síntoma | Solución |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | Se está usando otro Python. Ejecuta con `.venv\Scripts\python.exe` o, en VS Code, elige ese intérprete con *Python: Select Interpreter* |
| `The function is not implemented ... cvNamedWindow` al abrir la cámara | FiftyOne instala `opencv-python-headless`, que no tiene ventanas y pisa a `opencv-python`. Ejecuta `Iniciar_Demo_EPP.bat` (lo repara solo) o las dos últimas líneas de la sección 2.2 |
| La ventana del `.bat` dice `No se pudo abrir la fuente de video: 0` | La cámara está ocupada por otra aplicación o no hay cámara. Ciérrala o prueba con otro índice: `.\Iniciar_Demo_EPP.bat 1` |
| `Activate.ps1 ... la ejecución de scripts está deshabilitada` | No hace falta activar: usa `.venv\Scripts\python.exe` directamente |
| `No Python at '...AppData\Roaming\uv\python...'` | El Python de uv quedó en una carpeta no visible. Reinstálalo con `UV_PYTHON_INSTALL_DIR` apuntando al proyecto (sección 2) |
| `CUDA error` / `GPU is lost` al entrenar | Reinicia el equipo y revisa el controlador, la temperatura y la fuente. Mientras tanto usa `--dispositivo cpu` |
| La App de FiftyOne no muestra imágenes | El dataset guarda rutas absolutas. Si moviste la carpeta, bórralo con `fo.delete_dataset("epp-construction-ppe")` y vuelve a ejecutar `02_explorar_fiftyone.py` |
