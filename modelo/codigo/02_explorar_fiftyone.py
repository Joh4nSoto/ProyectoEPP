"""Paso 2: carga el dataset en FiftyOne y lo explora (sigue la guía de Voxel51).

Guía: https://docs.voxel51.com/getting_started/manufacturing/10_ppe_detection.html

Qué hace:
  1. Importa las imágenes y etiquetas YOLO a un dataset de FiftyOne.
  2. Calcula embeddings de cada caja (patch) con MobileNet-v2 del Model Zoo.
  3. Proyecta esos embeddings a 2D (UMAP) para ver qué clases se parecen o confunden.
  4. Abre la App de FiftyOne en el navegador (http://localhost:5151).

Uso:
    python modelo/codigo/02_explorar_fiftyone.py              # importa, calcula embeddings y abre la App
    python modelo/codigo/02_explorar_fiftyone.py --sin-app    # solo importa y calcula
    python modelo/codigo/02_explorar_fiftyone.py --rapido     # sin embeddings (más rápido)
"""
import argparse

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz

from config import DATASET_YAML

NOMBRE = "epp-construction-ppe"


def cargar_dataset():
    if fo.dataset_exists(NOMBRE):
        return fo.load_dataset(NOMBRE)
    dataset = fo.Dataset(NOMBRE)
    for split in ("train", "val"):
        dataset.add_dir(
            dataset_type=fo.types.YOLOv5Dataset,
            yaml_path=str(DATASET_YAML),
            split=split,
            tags=split,
            label_field="ground_truth",
        )
    dataset.persistent = True  # se conserva entre sesiones
    return dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sin-app", action="store_true")
    p.add_argument("--rapido", action="store_true")
    p.add_argument("--dispositivo", default="cpu", help="cpu o cuda para calcular embeddings")
    args = p.parse_args()

    dataset = cargar_dataset()
    print(dataset)
    print("\nCantidad de cajas por clase:")
    for clase, n in sorted(dataset.count_values("ground_truth.detections.label").items(), key=lambda x: -x[1]):
        print(f"  {clase:<10} {n}")

    calculados = dataset.count("ground_truth.detections.patches_embedding")
    total = dataset.count("ground_truth.detections")
    if not args.rapido and calculados < total:
        modelo = foz.load_zoo_model("mobilenet-v2-imagenet-torch", device=args.dispositivo)
        dataset.compute_patch_embeddings(
            model=modelo,
            patches_field="ground_truth",
            embeddings_field="patches_embedding",
            num_workers=0,
        )
        fob.compute_visualization(
            dataset,
            patches_field="ground_truth",
            embeddings="patches_embedding",
            brain_key="detection_patch_embeddings",
            num_workers=0,
        )

    if not args.sin_app:
        session = fo.launch_app(dataset, port=5151, auto=False)
        print(f"\nApp de FiftyOne en {session.url}  (Ctrl+C para cerrar)")
        session.wait()


if __name__ == "__main__":
    main()
