"""Lógica central del detector de EPP: carga el modelo YOLO, detecta y genera alertas.

Se usa desde 05_detectar_alertas.py, pero puede importarse en cualquier otro script:

    from modelo import DetectorEPP
    detector = DetectorEPP()
    resultado = detector.analizar("images/foto.jpg")
    print(resultado["alertas"])
"""
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import CONF_MINIMA, INCUMPLIMIENTOS, MODELO_ENTRENADO

COLOR_OK = (40, 170, 40)      # verde (BGR)
COLOR_ALERTA = (30, 30, 220)  # rojo
COLOR_OTRO = (200, 140, 30)   # azul


class DetectorEPP:
    def __init__(self, pesos=MODELO_ENTRENADO, conf=CONF_MINIMA):
        pesos = Path(pesos)
        if not pesos.exists():
            raise FileNotFoundError(f"No existe {pesos}. Ejecuta primero modelo/codigo/03_entrenar_yolo.py")
        self.modelo = YOLO(str(pesos))
        self.conf = conf

    def detectar(self, imagen):
        """Devuelve la lista de detecciones de una imagen (ruta o arreglo numpy BGR)."""
        r = self.modelo.predict(imagen, conf=self.conf, verbose=False)[0]
        detecciones = []
        for caja, clase, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            detecciones.append({
                "clase": r.names[int(clase)],
                "confianza": round(conf, 3),
                "caja": [round(v) for v in caja],
            })
        return detecciones, r.orig_img

    @staticmethod
    def evaluar_reglas(detecciones):
        """Aplica las reglas de cumplimiento y devuelve las alertas generadas."""
        alertas = []
        for d in detecciones:
            if d["clase"] in INCUMPLIMIENTOS:
                severidad, mensaje = INCUMPLIMIENTOS[d["clase"]]
                alertas.append({"severidad": severidad, "mensaje": mensaje,
                                "clase": d["clase"], "confianza": d["confianza"]})
        return alertas

    @staticmethod
    def dibujar(img, detecciones, alertas):
        img = img.copy()
        esc = max(img.shape[1] / 1000, 0.5)  # escala del texto según el ancho de la imagen
        grosor = max(int(2 * esc), 1)
        for d in detecciones:
            x1, y1, x2, y2 = d["caja"]
            if d["clase"] in INCUMPLIMIENTOS:
                color = COLOR_ALERTA
            elif d["clase"] in ("Person", "forklift"):
                color = COLOR_OTRO
            else:
                color = COLOR_OK
            cv2.rectangle(img, (x1, y1), (x2, y2), color, grosor)
            etiqueta = f"{d['clase']} {d['confianza']:.2f}"
            (w, h), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * esc, grosor)
            cv2.rectangle(img, (x1, y1 - h - 8), (x1 + w + 4, y1), color, -1)
            cv2.putText(img, etiqueta, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * esc, (255, 255, 255), grosor)
        # Banda superior con el estado general
        estado = f"ALERTA: {len(alertas)} incumplimiento(s)" if alertas else "OK: EPP completo"
        alto = int(40 * esc)
        cv2.rectangle(img, (0, 0), (img.shape[1], alto), COLOR_ALERTA if alertas else COLOR_OK, -1)
        cv2.putText(img, estado, (10, int(alto * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * esc, (255, 255, 255), grosor + 1)
        return img

    def analizar(self, imagen, origen=None):
        detecciones, img = self.detectar(imagen)
        alertas = self.evaluar_reglas(detecciones)
        return {
            "origen": origen or (str(imagen) if isinstance(imagen, (str, Path)) else "frame"),
            "fecha": datetime.now().isoformat(timespec="seconds"),
            "personas": sum(d["clase"] == "Person" for d in detecciones),
            "detecciones": detecciones,
            "alertas": alertas,
            "imagen_anotada": self.dibujar(img, detecciones, alertas),
        }
