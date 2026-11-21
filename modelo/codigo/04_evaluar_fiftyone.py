"""Paso 4: evalúa el modelo entrenado sobre el conjunto de validación con FiftyOne.

Agrega las predicciones de YOLO al dataset, calcula precisión, recall y mAP por
clase, y guarda el informe en resultados/evaluacion.txt. Con --app se pueden
revisar en la App los falsos positivos y falsos negativos imagen por imagen.

Uso:
    python modelo/codigo/04_evaluar_fiftyone.py
    python modelo/codigo/04_evaluar_fiftyone.py --app
"""
import argparse
import contextlib
import io
import json

import fiftyone as fo
from fiftyone import ViewField as F
from ultralytics import YOLO

from config import CLASES, CONF_MINIMA, INCUMPLIMIENTOS, MODELO_ENTRENADO, RESULTADOS

NOMBRE = "epp-construction-ppe"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--app", action="store_true")
    args = p.parse_args()

    if not fo.dataset_exists(NOMBRE):
        raise SystemExit("Primero ejecuta modelo/codigo/02_explorar_fiftyone.py para crear el dataset.")
    dataset = fo.load_dataset(NOMBRE)
    val = dataset.match_tags("val")

    modelo = YOLO(str(MODELO_ENTRENADO))
    val.apply_model(modelo, label_field="prediccion", confidence_thresh=CONF_MINIMA)

    res = val.evaluate_detections("prediccion", gt_field="ground_truth", eval_key="eval", compute_mAP=True)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res.print_report(classes=CLASES)
    reporte = buf.getvalue()
    resumen = {
        "imagenes_validacion": len(val),
        "mAP_COCO_0.50_0.95": round(res.mAP(), 4),
        "verdaderos_positivos": int(sum(val.values("eval_tp"))),
        "falsos_positivos": int(sum(val.values("eval_fp"))),
        "falsos_negativos": int(sum(val.values("eval_fn"))),
    }
    # Recall de las clases que disparan alertas (requisito del plan: recall >= 0,95 a futuro)
    metricas = res.report(classes=list(INCUMPLIMIENTOS))
    resumen["recall_clases_alerta"] = {c: round(metricas[c]["recall"], 3) for c in INCUMPLIMIENTOS if c in metricas}

    RESULTADOS.mkdir(exist_ok=True)
    (RESULTADOS / "evaluacion.txt").write_text(reporte + "\n" + json.dumps(resumen, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    print(reporte)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))

    if args.app:
        # Vista ordenada por imágenes con más falsos negativos (EPP faltante no detectado)
        vista = val.sort_by("eval_fn", reverse=True).filter_labels("prediccion", F("confidence") > CONF_MINIMA)
        session = fo.launch_app(vista, port=5151, auto=False)
        print(f"App de FiftyOne en {session.url}  (Ctrl+C para cerrar)")
        session.wait()


if __name__ == "__main__":
    main()
