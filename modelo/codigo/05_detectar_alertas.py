"""Paso 5: demo de detección de EPP con alertas.

Analiza imágenes, un video o la cámara web, dibuja las detecciones y genera
una alerta cuando detecta un incumplimiento (por ejemplo, persona sin casco).

Uso:
    python modelo/codigo/05_detectar_alertas.py                       # todas las imágenes de images/
    python modelo/codigo/05_detectar_alertas.py --fuente foto.jpg     # una imagen
    python modelo/codigo/05_detectar_alertas.py --fuente video.mp4    # un video
    python modelo/codigo/05_detectar_alertas.py --fuente 0            # cámara web (q para salir)

Salidas en resultados/: imágenes anotadas y alertas.csv con el registro de alertas.
Para cumplir el plan, el sistema NO identifica personas: solo registra la omisión de EPP.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2

from config import IMAGENES, RESULTADOS
from modelo import DetectorEPP

EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEGUNDOS_PERSISTENCIA = 3  # requisito del plan: alertar si la omisión dura >= 3 s en video


def registrar(alertas_csv, resultado):
    nuevo = not alertas_csv.exists()
    with open(alertas_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["fecha", "origen", "severidad", "mensaje", "clase", "confianza"])
        for a in resultado["alertas"]:
            w.writerow([resultado["fecha"], resultado["origen"], a["severidad"], a["mensaje"],
                        a["clase"], a["confianza"]])


def imprimir(resultado):
    nombre = Path(resultado["origen"]).name
    if resultado["alertas"]:
        print(f"[ALERTA] {nombre}: " + "; ".join(
            f"{a['mensaje']} ({a['severidad']}, conf {a['confianza']:.2f})" for a in resultado["alertas"]))
    else:
        print(f"[OK]     {nombre}: {resultado['personas']} persona(s), sin incumplimientos")


def procesar_imagenes(detector, rutas, alertas_csv):
    salida = RESULTADOS / "imagenes"
    salida.mkdir(parents=True, exist_ok=True)
    total_alertas = 0
    for ruta in rutas:
        r = detector.analizar(str(ruta))
        cv2.imwrite(str(salida / ruta.name), r["imagen_anotada"])
        imprimir(r)
        registrar(alertas_csv, r)
        total_alertas += bool(r["alertas"])
    print(f"\n{len(rutas)} imágenes analizadas, {total_alertas} con incumplimientos.")
    print(f"Imágenes anotadas en {salida}")


def procesar_video(detector, fuente, alertas_csv, mostrar):
    cap = cv2.VideoCapture(int(fuente) if fuente.isdigit() else fuente)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente de video: {fuente}")
    escritor = None
    inicio_omision = {}  # clase -> instante en que empezó la omisión
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = detector.analizar(frame, origen=f"video:{fuente}")
        ahora = time.time()
        activas = {a["clase"] for a in r["alertas"]}
        for clase in list(inicio_omision):
            if clase not in activas:
                del inicio_omision[clase]
        confirmadas = []
        for a in r["alertas"]:
            t0 = inicio_omision.setdefault(a["clase"], ahora)
            if ahora - t0 >= SEGUNDOS_PERSISTENCIA:
                confirmadas.append(a)
                inicio_omision[a["clase"]] = ahora + 30  # evita repetir la alerta por 30 s
        if confirmadas:
            r["alertas"] = confirmadas
            imprimir(r)
            registrar(alertas_csv, r)
        img = r["imagen_anotada"]
        if escritor is None and not fuente.isdigit():
            (RESULTADOS / "video").mkdir(parents=True, exist_ok=True)
            h, w = img.shape[:2]
            escritor = cv2.VideoWriter(str(RESULTADOS / "video" / (Path(fuente).stem + "_anotado.mp4")),
                                       cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS) or 25, (w, h))
        if escritor:
            escritor.write(img)
        if mostrar:
            cv2.imshow("Deteccion de EPP (q para salir)", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if escritor:
        escritor.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fuente", default=str(IMAGENES), help="carpeta, imagen, video o índice de cámara")
    p.add_argument("--pesos", default=None, help="ruta a otros pesos .pt (opcional)")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--sin-ventana", action="store_true", help="no mostrar ventana en modo video")
    args = p.parse_args()

    kwargs = {}
    if args.pesos:
        kwargs["pesos"] = args.pesos
    if args.conf is not None:
        kwargs["conf"] = args.conf
    detector = DetectorEPP(**kwargs)
    RESULTADOS.mkdir(exist_ok=True)
    alertas_csv = RESULTADOS / "alertas.csv"

    fuente = Path(args.fuente)
    if fuente.is_dir():
        rutas = sorted(x for x in fuente.iterdir() if x.suffix.lower() in EXT_IMG)
        procesar_imagenes(detector, rutas, alertas_csv)
    elif fuente.suffix.lower() in EXT_IMG:
        procesar_imagenes(detector, [fuente], alertas_csv)
    else:
        procesar_video(detector, args.fuente, alertas_csv, mostrar=not args.sin_ventana)
    print(f"Registro de alertas: {alertas_csv}")


if __name__ == "__main__":
    main()
