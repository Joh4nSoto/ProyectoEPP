"""Paso 3: entrena (fine-tuning) un modelo YOLO11n para detectar EPP.

Parte de pesos preentrenados en COCO y los ajusta al dataset Construction-PPE
(transfer learning). Con una RTX 3060 Ti cada época toma ~35 s (30 épocas, ~20 min); en CPU, ~4 min por época.

Uso:
    python modelo/codigo/03_entrenar_yolo.py            # 30 épocas por defecto
    python modelo/codigo/03_entrenar_yolo.py --epocas 60 --modelo yolo11s.pt
    python modelo/codigo/03_entrenar_yolo.py --dispositivo cpu --modelo epp_yolo11n.pt --epocas 10  # seguir en CPU
"""
import argparse
import shutil

import torch
from ultralytics import YOLO

from config import DATASET_YAML, MODELO_BASE, MODELO_ENTRENADO, MODELOS, RUNS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epocas", type=int, default=30)
    p.add_argument("--modelo", default=MODELO_BASE, help="pesos de partida (yolo11n.pt, yolo11s.pt, ...)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--dispositivo", default="auto", help="auto, cpu o 0 (primera GPU)")
    args = p.parse_args()

    if args.dispositivo == "auto":
        dispositivo = 0 if torch.cuda.is_available() else "cpu"
    else:
        dispositivo = int(args.dispositivo) if args.dispositivo.isdigit() else args.dispositivo
    print(f"Entrenando en: {'GPU ' + torch.cuda.get_device_name(0) if dispositivo == 0 else 'CPU'}")

    MODELOS.mkdir(exist_ok=True)
    # --modelo acepta un nombre (yolo11n.pt), un archivo en pesos/ o una ruta completa
    ruta = MODELOS / args.modelo
    modelo = YOLO(str(ruta) if ruta.exists() else args.modelo)
    modelo.train(
        data=str(DATASET_YAML),
        epochs=args.epocas,
        imgsz=args.imgsz,
        batch=args.batch,
        device=dispositivo,
        workers=0,           # en Windows evita problemas de multiprocesamiento
        project=str(RUNS),
        name="entrenamiento_epp",
        exist_ok=True,
        seed=42,
        plots=True,
    )

    mejor = RUNS / "entrenamiento_epp" / "weights" / "best.pt"
    shutil.copy(mejor, MODELO_ENTRENADO)
    print(f"Modelo final copiado a {MODELO_ENTRENADO}")

    metricas = YOLO(str(MODELO_ENTRENADO)).val(data=str(DATASET_YAML), device=dispositivo,
                                                 workers=0, project=str(RUNS), name="validacion", exist_ok=True)
    print(f"mAP50 = {metricas.box.map50:.3f}   mAP50-95 = {metricas.box.map:.3f}")


if __name__ == "__main__":
    main()
