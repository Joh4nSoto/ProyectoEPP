"""Configuración compartida del prototipo de detección de EPP.

Todas las rutas se calculan a partir de la carpeta del modelo (E:\\ProyectoEPP\\modelo),
así los scripts funcionan sin importar desde qué carpeta se ejecuten.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

DATOS = RAIZ / "datos"
DATASET = DATOS / "construction-ppe"          # dataset descargado por 01_descargar_datos.py
DATASET_YAML = DATOS / "construction-ppe.yaml"  # archivo de configuración que lee YOLO
DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"

MODELOS = RAIZ / "pesos"
MODELO_ENTRENADO = MODELOS / "epp_yolo11n.pt"   # pesos que deja 03_entrenar_yolo.py
MODELO_BASE = "yolo11n.pt"                       # modelo preentrenado en COCO (se descarga solo)

IMAGENES = RAIZ / "images"         # imágenes de prueba para la demo
RESULTADOS = RAIZ / "resultados"   # imágenes anotadas, alertas y métricas
RUNS = RAIZ / "runs"               # carpeta de trabajo de Ultralytics

# Clases del dataset Construction-PPE (Ultralytics, CC BY 4.0).
# Las clases "no_*" marcan explícitamente la AUSENCIA del EPP, lo que permite
# generar alertas sin reglas geométricas complejas.
CLASES = [
    "helmet", "gloves", "vest", "boots", "goggles", "Person",
    "no_helmet", "no_goggle", "no_gloves", "no_boots", "forklift", "no_vest",
]

# Reglas de incumplimiento: clase detectada -> mensaje de alerta.
# El plan prioriza casco y chaleco (Must); guantes, botas y antiparras quedan como Should.
INCUMPLIMIENTOS = {
    "no_helmet": ("ALTA", "Trabajador sin casco"),
    "no_vest": ("ALTA", "Trabajador sin chaleco reflectante"),
    "no_gloves": ("MEDIA", "Trabajador sin guantes"),
    "no_boots": ("MEDIA", "Trabajador sin calzado de seguridad"),
    "no_goggle": ("MEDIA", "Trabajador sin antiparras"),
}

CONF_MINIMA = 0.35  # confianza mínima para considerar una detección
