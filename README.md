# 🚀 System180 Demonstrator

Schneller Einstieg:

```bash
git clone https://github.com/ptrstn/system180-demonstrator
cd system180-demonstrator
./model-conversion.sh
python main.py
```

> Alternativ (für Produktionsumgebung):
>
> ```bash
> uvicorn main:app --host 0.0.0.0 --port 8000
> ```

🔗 Danach einfach im Browser öffnen: [http://localhost:8000](http://localhost:8000)

---

## 📦 YOLO TensorRT Modelle

| Modell-Datei                         | YOLO-Version | Task         | Auflösung | Präzision | Klassen | Beschreibung                             |
| ------------------------------------ | ------------ | ------------ | --------- | --------- | ------- | ---------------------------------------- |
| `custom_320_FP16_detect.engine`      | YOLOv11s     | Detection    | 320×320   | FP16      | 26      | Hauptmodell für reale Defekte            |
| `synthetic_320_FP16_detect.engine`   | YOLOv11l     | Detection    | 320×320   | FP16      | 29      | Ergänzungsmodell mit synthetischen Daten |
| `NubsUpDown_320_FP16_segment.engine` | YOLOv11s-seg | Segmentation | 320×320   | FP16      | 2       | Nubs-Status-Erkennung (oben/unten)       |

---

## 🎥 Kamera-Setup & Modell-Zuweisung

| Position   | Kameratyp          | Zugewiesene Modelle                                                  | Hinweise                                                |
| ---------- | ------------------ | -------------------------------------------------------------------- | ------------------------------------------------------- |
| **Links**  | OAK-1 Max          | `NubsUpDown_320_FP16_segment.engine`                                 | Segmentierung von Nubs direkt auf der Kamera            |
| **Rechts** | OAK-1 Max          | `NubsUpDown_320_FP16_segment.engine`                                 | Identisch zum linken Setup                              |
| **Mitte**  | OBSBOT Meet 2 (4K) | `custom_320_FP16_detect.engine` + `synthetic_320_FP16_detect.engine` | Ensemble-Detection über OpenCV-Inferenz + ArUco-Messung |

---

## 💡 Hinweise zur Architektur

* **OAK-1 Max** Kameras verwenden DepthAI + TensorRT auf dem Gerät selbst (inkl. Polygon-ROI).
* **OBSBOT Webcam** läuft über OpenCV auf dem Jetson Orin Nano mit Echtzeit-YOLO-Inferenz (2 Modelle gleichzeitig).
* Die mittlere Kamera misst zusätzlich mithilfe von **ArUco-Markern** Objektgröße und Position.

---

## 🔗 Nützliche Links & Tools

* **Luxonis Deployment Guides**
  [Jetson Deployment Guide](https://docs.luxonis.com/hardware/platform/deploy/to-jetson/)
* **Blob-Konvertierung für OAK**
  [blobconverter.luxonis.com](https://blobconverter.luxonis.com/)
* **Ultralytics YOLO Dokumentation**
  [docs.ultralytics.com](https://docs.ultralytics.com/)

