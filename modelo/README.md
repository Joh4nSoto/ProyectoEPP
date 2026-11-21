# Prototipo de detección de EPP con YOLO y FiftyOne

**Proyecto:** Sistema de vigilancia inteligente para el uso correcto de Elementos de Protección Personal (EPP)
**Equipo:** Johan Soto y Fernanda Alcaino

Este prototipo detecta EPP en imágenes, video o cámara web, y genera una **alerta** cuando un
trabajador no usa casco, chaleco, guantes, botas o antiparras. Corre 100 % local, sin internet
después de la instalación, como exige el plan (RNF de disponibilidad sin conexión).

Sigue el flujo de la guía de Voxel51
[PPE Detection and Safety Monitoring](https://docs.voxel51.com/getting_started/manufacturing/10_ppe_detection.html):
cargar un dataset YOLO en FiftyOne, explorarlo con embeddings y, como paso agregado, entrenar
y evaluar un detector YOLO de Ultralytics.

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

Se usa [uv](https://docs.astral.sh/uv/) para crear el entorno con Python 3.11.

```powershell
cd E:\ProyectoEPP
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"   # Python dentro del proyecto, no en AppData
uv python install 3.11
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv\Scripts\python.exe -r modelo\codigo\requirements.txt
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

## 6. Privacidad y límites

- El sistema no identifica personas ni guarda rostros por separado: registra solo fecha, origen,
  tipo de incumplimiento y confianza (`alertas.csv`), en línea con las exclusiones del plan.
- El modelo se entrenó con 800 imágenes de obras de construcción. En una planta real habrá que
  reentrenar con imágenes propias etiquetadas (fase de preparación de datos del plan).
- Las métricas de este prototipo no alcanzan aún la meta de recall ≥ 0,95; ver
  `modelo/Registro_Ejecucion_Prototipo_EPP.md`.
