# Sistema de monitoreo de EPP con visión artificial

Sistema de vigilancia inteligente basado en cámaras y visión artificial que detecta el uso
incorrecto de los **Elementos de Protección Personal (EPP)** en el entorno laboral y genera
alertas para prevenir accidentes. El proyecto evalúa su viabilidad en tres dimensiones:
**ética**, **financiera** y de **gestión de riesgos**, e incluye un prototipo funcional con YOLO.

**Asignatura:** Gestión de Proyectos de Datos (SCY1102)
**Equipo:** Johan Soto y Fernanda Alcaino

| | Enlace |
|---|---|
| **Plan de proyecto (v4), documento principal** | [`Plan_de_Proyecto_EPP_v4.pdf`](Plan_de_Proyecto_EPP_v4.pdf) |
| Anexos A–F del plan | [`entregables/Plan_v4/`](entregables/Plan_v4/) |
| Arquitectura | [`entregables/Arquitectura_Sistema_EPP_v1.2.pdf`](entregables/Arquitectura_Sistema_EPP_v1.2.pdf) |
| Modelo: guía de uso y de YOLO | [`modelo/README.md`](modelo/README.md) |
| Registro de ejecución del prototipo | [`modelo/Registro_Ejecucion_Prototipo_EPP.md`](modelo/Registro_Ejecucion_Prototipo_EPP.md) |
| Tablero de seguimiento (Jira) | [Proyecto ME en Jira](https://sistema-monitoreo-epp.atlassian.net/jira/core/projects/ME/board) |

---

## Estructura del repositorio

```
ProyectoEPP/
├── Plan_de_Proyecto_EPP_v4.pdf   documento principal del proyecto
├── README.md                     este archivo
├── entregables/
│   ├── Plan_v4/                  anexos A–F del plan (PDF)
│   ├── editables/                versiones Word de los documentos vigentes
│   ├── arquitectura_diagramas/   figuras de la arquitectura
│   ├── presentacion_imagenes/    gráficos para la presentación
│   └── Arquitectura_Sistema_EPP_v1.2.pdf
├── modelo/                       todo lo del prototipo de detección
│   ├── README.md                 instalación, uso y explicación de YOLO
│   ├── Registro_Ejecucion_Prototipo_EPP.md
│   ├── codigo/                   scripts en Python (YOLO + FiftyOne)
│   ├── pesos/epp_yolo11n.pt      modelo entrenado (5,5 MB)
│   ├── images/                   12 imágenes de prueba
│   └── evidencia/                capturas, métricas y alertas
└── documentacion/                informes de factibilidad previos (GP1 y GP2)
```

No se versionan el entorno de Python (`.venv`, `.python`), el dataset (`modelo/datos/`,
~620 MB), las salidas de entrenamiento (`modelo/runs/`) ni las de la demo
(`modelo/resultados/`). Todo se regenera con los scripts de `modelo/codigo/`.

## Ejecutar la demo

Requisitos: Windows, [uv](https://docs.astral.sh/uv/) y, opcionalmente, una GPU NVIDIA.

```powershell
git clone https://github.com/Joh4nSoto/ProyectoEPP.git
cd ProyectoEPP
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"
uv python install 3.11
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv\Scripts\python.exe -r modelo\codigo\requirements.txt
.venv\Scripts\python.exe modelo\codigo\05_detectar_alertas.py
```

La demo analiza las imágenes de `modelo/images/` con el modelo ya entrenado y deja en
`modelo/resultados/` las imágenes anotadas y el registro `alertas.csv`. Otras fuentes:

```powershell
.venv\Scripts\python.exe modelo\codigo\05_detectar_alertas.py --fuente C:\ruta\video.mp4
.venv\Scripts\python.exe modelo\codigo\05_detectar_alertas.py --fuente 0      # cámara web, q para salir
```

Para reentrenar el modelo o explorar los datos en FiftyOne, ver [`modelo/README.md`](modelo/README.md).

## Resultados del prototipo

Modelo YOLO11n ajustado con el dataset público Construction-PPE (1.000 imágenes):
precisión global 0,91, recall 0,70 y mAP50 0,73. El recall de las clases de incumplimiento
(sin casco 0,50; sin chaleco 0,58) aún no alcanza la meta del plan (≥ 0,95), que requiere el
dataset propio de la planta. Detalle en el
[registro de ejecución](modelo/Registro_Ejecucion_Prototipo_EPP.md).

## Licencias de terceros

- Dataset Construction-PPE de Ultralytics, licencia CC BY 4.0. Las imágenes de `modelo/images/` provienen de él.
- Ultralytics YOLO, licencia AGPL-3.0.
- FiftyOne de Voxel51, licencia Apache 2.0.
