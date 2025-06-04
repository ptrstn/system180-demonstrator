## 1. Ensemble-Prediction (Webcam)

Im alten Code existierte eine Funktion `ensemble_predictions`, die zwei YOLO-Ergebnisse per Non-Maximum-Suppression kombiniert, um im Idealfall verbesserte oder robustere Detektionen zu erhalten (z. B. zwei Modelle gleichzeitig auf die Webcam-ROI anwenden). Diese Methode wurde in der neuen, performanten Version noch nicht eingebaut.

**Wo das steht:**

```python
def ensemble_predictions(results1, results2, iou_threshold=0.5):
    # Kombiniert Ergebnisse von results1 & results2 per NMS
    …
```



Wenn du „Ensemble-Prediction“ wieder aktivieren möchtest, müssten wir:

1. Beide YOLO-Modelle (z. B. ein Hauptmodell und ein zweites Modell) für die Webcam laden.
2. Im `DetectCamera`-Loop oder einer separaten Klasse die beiden Inferenz-Ergebnisse zusammenführen, bevor wir Bounding-Boxes zeichnen.

---

## 2. Polygonale und rechteckige ROIs

Im alten Code gab es eine Klasse `PolygonROI`, mit der man frei definierbare Polygone als ROI über das Bild legen konnte („Maske erstellen“ → Extrahieren des ROI → Anzeige). Außerdem wurden affin definierte rechteckige ROIs (`self.rect_rois`) für jede Kamera-Videoframe verwendet. Aktuell im neuen „main.py“ arbeiten wir nur auf dem vollen 320×320-Crop, ohne polygonale ROI-Extraktion oder blaues Rechteck zum Aufzeichnen von ROI-Bereichen.

**Wo das steht:**

```python
class PolygonROI:
    def __init__(self, points):
        self.points = points
        self.mask = None
        …
    def extract_roi(self, frame):
        # ROI aus dem Frame ausschneiden
        …
```



Und die Definition der Rechteck-ROIs:

```python
self.rect_rois = {
    "webcam": (10, 90, 620, 160),
    "left": (80, 20, 440, 390),
    "right": (80, 20, 440, 390)
}
```



Falls du z. B. nur innerhalb einer polygonalen Maske detektieren möchtest (statt auf das volle 320×320 Bild), musst du:

1. `PolygonROI` initialisieren (z. B. mit Eckpunkten).
2. Bei jedem Frame mit `extract_roi(...)` den Masken-ROI ausschneiden.
3. Nur in dieser ROI YOLO ausführen (oder maskiertes Ergebnis wieder zurück in den Full-Frame projizieren).

---

## 3. Save-Funktionalitäten („save\_for\_roboflow“ / „save\_detections“)

Im Original gab es Methoden, um Detektionen als Bilder und Label-Txts für Roboflow zu exportieren („save\_for\_roboflow“), sowie einen FastAPI-Endpoint `/save-detections`, der die aktuell angezeigten Detektionen in eine CSV-Datei schreibt (`save_detections_to_dataframe`).

**Wo das steht:**

```python
def save_for_roboflow(self, frame, boxes, classes, roi_type, frame_count):
    # speichert ROI als Bild und YOLO-TXT-Labels für Roboflow
    …
  
@app.post("/save-detections")
async def save_detections():
    result = camera_system.save_detections_to_dataframe()
    return result
```



Wenn du wieder eine ähnliche Funktion benötigst, müssten wir:

1. Einen internen Speicher (Liste) für Detektions-Daten in der neuen Struktur ergänzen.
2. Einen Endpoint zum Abrufen/Speichern dieser Liste in eine CSV implementieren.

---

## 4. WebSocket-Support

Im alten UI gab es einen `/ws`-WebSocket, über den Nachrichten an und vom Frontend geschickt wurden. Aktuell funktionieren wir nur mit MJPEG-Streams, ohne WebSocket-Kanäle.

**Wo das steht:**

```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    …
    await websocket.send_json({"status":"success","message":"System läuft."})
```



Wenn du diesen WebSocket-Kanal wiederhaben möchtest, um z. B. laufende Statusmeldungen oder Anzeigen aus dem Backend per WebSocket an das Frontend zu pushen, müssten wir:

1. Die WebSocket-Routen importieren/integrieren.
2. In den Camera-Threads bei Bedarf Status-Updates senden (z. B. Detektionscounts, neue Ratio-Werte, Fehlermeldungen).

---

## 5. Anzeige von FPS und Status‐Infos direkt im Frame

Im alten Code wurde in der Hauptschleife die aktuelle Framerate berechnet und per `cv2.putText(...)` in die Ecke des Frames geschrieben (u. a. Warnung, falls kein ArUco gefunden etc.). Im neuen `main.py` zeigen wir nur eine einfache Frame-Nummer an, aber keine FPS-Anzeige.

**Wo das steht:**

```python
fps_text = f"FPS: {int(1/(time.time()-last_time))}"
cv2.putText(display_frame, fps_text, (10, height-10), …)
```



Wenn du die FPS-Anzeige (und z. B. Warnungen „Kein ArUco erkannt“) wiederhaben möchtest, reicht es aus, anstelle der reinen Frame-Nummer in `DetectCamera._inference_loop()`:

```python
# Beispiel:
fps = 1.0 / max((time.time() - self._last_time), 1e-3)
self._last_time = time.time()
cv2.putText(annotated_320, f"FPS: {fps:.1f}", (5, self.model_input_size - 5), …)
```

zu ergänzen. Dann siehst du direkt im Stream, wie schnell Inferenz + Postprocessing tatsächlich laufen.

---

## 6. Platzhalter‐Frames, falls eine Kamera fehlt

Das alte UI hatte eine Methode `create_placeholder_frame(text, width, height)`, die ein schwarzes Bild mit einem Hinweis‐Text erzeugt, falls z. B. eine Kamera nicht angeschlossen ist. Im neuen Setup starten wir allerdings die App erst, wenn alle Kameras verbunden sind, sodass wir noch keine „Platzhalter“ anlegen.

**Wo das steht:**

```python
def create_placeholder_frame(self, text, width, height):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, text, …)
    return frame
```



Falls du weiterhin „soft fail“ möchtest (d. h. bei fehlender OAK-Kamera trotzdem UI zeigen, nur mit einem Hinweisschirm), könntest du in `list_oak_devices()` anstelle von Exception eine Placeholder-Kamera‐Klasse instanziieren, die permanent das Platzhalterbild liefert.

---

### Zusammenfassung

Bislang fehlen in „main.py“ also (mindestens) diese Features aus dem alten Code:

1. **Ensemble-Prediction** (für Webcam).
2. **Polygonale/rectangular ROIs** (und Masken-Overlay).
3. **Save-Funktionalität** (Roboflow/CSV).
4. **WebSocket-Support** (Statusmeldungen).
5. **FPS-Anzeige & weitere Statuswarnungen** direkt im Frame.
6. **Platzhalter-Frames**, falls Kameras nicht verfügbar.

---

**Vorschlag für den nächsten Schritt:**
Wenn Du Schritt für Schritt testen möchtest, würde ich empfehlen, zuerst die **FPS-Anzeige** (Punkt 5) ins neue `main.py` zu integrieren. Das ist relativ überschaubar, bricht keine bereits implementierte Pipeline und erlaubt Dir sofort eine visuelle Rückmeldung, ob die ArUco-/YOLO-Schleife tatsächlich wie gewünscht läuft.

Alternativ könntest Du auch direkt die **Ensemble-Prediction** (Punkt 1) aktivieren, falls Dir wichtig ist, zwei Modelle parallel laufen zu lassen. Das erfordert jedoch etwas mehr Anpassung (zweites Modell laden, Ergebnisse mergen).

Lass mich wissen, welchen dieser Punkte Du als Nächstes ergänzt haben möchtest, dann kann ich Dir den entsprechenden Code-Schnipsel liefern.
