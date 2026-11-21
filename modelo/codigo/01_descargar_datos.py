"""Paso 1: descarga el dataset Construction-PPE y genera el YAML para YOLO.

Uso:
    python modelo/codigo/01_descargar_datos.py
"""
import shutil
import urllib.request
import zipfile

from config import CLASES, DATASET, DATASET_URL, DATASET_YAML, DATOS, IMAGENES


def descargar():
    if (DATASET / "images" / "train").exists():
        print(f"El dataset ya existe en {DATASET}")
        return
    DATOS.mkdir(parents=True, exist_ok=True)
    destino_zip = DATOS / "construction-ppe.zip"
    print(f"Descargando {DATASET_URL} (~650 MB)...")
    urllib.request.urlretrieve(DATASET_URL, destino_zip)
    print("Descomprimiendo...")
    with zipfile.ZipFile(destino_zip) as z:
        z.extractall(DATOS)
    destino_zip.unlink()


def escribir_yaml():
    lineas = [f"path: {DATASET.as_posix()}", "train: images/train", "val: images/val", "names:"]
    lineas += [f"  {i}: {c}" for i, c in enumerate(CLASES)]
    DATASET_YAML.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"YAML escrito en {DATASET_YAML}")


def copiar_imagenes_demo(n=12):
    """Copia algunas imágenes de validación a images/ para probar la demo."""
    IMAGENES.mkdir(exist_ok=True)
    if any(IMAGENES.iterdir()):
        return
    for img in sorted((DATASET / "images" / "val").glob("*.jpg"))[:n]:
        shutil.copy(img, IMAGENES / img.name)
    print(f"Copiadas {n} imágenes de prueba a {IMAGENES}")


if __name__ == "__main__":
    descargar()
    escribir_yaml()
    copiar_imagenes_demo()
    n_train = len(list((DATASET / "images" / "train").glob("*")))
    n_val = len(list((DATASET / "images" / "val").glob("*")))
    print(f"Listo: {n_train} imágenes de entrenamiento y {n_val} de validación.")
