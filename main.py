# main.py

import threading
import time
from typing import Generator, Optional

import cv2
import depthai as dai
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from aruco_utils import detect_aruco_markers_with_tracking, calculate_dimensions

# ============================================
# KAMEREN- / INFERENZ-KONSTANTEN
# ============================================

# OAK-1 Max (DepthAI) Kameraeinstellungen (LEFT & RIGHT segmentieren)
OAK_FRAME_WIDTH:        int   = 640
OAK_FRAME_HEIGHT:       int   = 480
OAK_FPS:                int   = 30
YOLO_ENGINE_LEFT:       str   = "./models/NubsUpDown_320_FP16_segment.engine"
YOLO_ENGINE_RIGHT:      str   = "./models/NubsUpDown_320_FP16_segment.engine"

# USB-Webcam (CENTER detektieren, Ensemble-Prediction)
USB_DEVICE_INDEX:       int   = 0
USB_CAPTURE_WIDTH:      int   = 640
USB_CAPTURE_HEIGHT:     int   = 480
USB_CAPTURE_FPS:        int   = 30
YOLO_ENGINE_CENTER_1:   str   = "./models/custom_320_FP16_detect.engine"
YOLO_ENGINE_CENTER_2:   str   = "./models/synthetic_320_FP16_detect.engine"

# YOLO-Inference-Grundeinstellungen
YOLO_MODEL_INPUT_SIZE:  int   = 320
YOLO_CONF_THRESH:       float = 0.25
YOLO_IOU_THRESH:        float = 0.45
YOLO_MAX_DET:           int   = 300
YOLO_DEVICE:            str   = "cuda"

# MJPEG-Boundary
MJPEG_BOUNDARY:         bytes = b"--frameboundary"

# Inferenz-Frequenz (skip N-1 Frames)
SKIP_FRAMES:            int   = 5

# Batch-Größe
BATCH_SIZE:             int   = 1

# ArUco-Einstellungen
ARUCO_ENABLED:          bool  = True    # True = ArUco ein, False = ArUco aus
ARUCO_SKIP_FRAMES:      int   = 5       # alle N Frames ArUco laufen lassen


# ============================================
# FASTAPI-Setup
# ============================================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ============================================
# Basisklasse: FrameGrabber
# ============================================
class FrameGrabber:
    """
    Basisklasse für eine Kamera, die im Hintergrund Frames abruft
    und immer nur das zuletzt gelesene Frame speichert.
    """
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._is_running = False

    def start(self) -> None:
        """
        Thread starten, der kontinuierlich Frames abruft.
        """
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Thread anhalten.
        """
        self._is_running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)

    def _capture_loop(self) -> None:
        """
        Muss von Unterklassen überschrieben werden. Setzt self._latest_frame.
        """
        raise NotImplementedError("Subclasses must implement _capture_loop")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Gibt das aktuellste Frame (BGR) zurück, oder None.
        """
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def _set_frame(self, frame: np.ndarray) -> None:
        """
        Intern: Speichert das neueste Frame.
        """
        with self._lock:
            self._latest_frame = frame


# ============================================
# OAK-1 Max Camera (nur Raw-Frames)
# ============================================
class OAK1MaxCamera(FrameGrabber):
    """
    Implementierung für OAK-1 Max (DepthAI). Liefert Vorschau-Frames in BGR.
    """
    def __init__(
        self,
        device_id: Optional[str],
        width: int = OAK_FRAME_WIDTH,
        height: int = OAK_FRAME_HEIGHT,
        fps: int = OAK_FPS
    ) -> None:
        super().__init__()
        self._device_id = device_id
        self._frame_width = width
        self._frame_height = height
        self._fps = fps
        self._pipeline: Optional[dai.Pipeline] = None
        self._device:   Optional[dai.Device] = None
        self._out_queue: Optional[dai.DataOutputQueue] = None
        self._initialize_depthai()

    def _initialize_depthai(self) -> None:
        pipeline = dai.Pipeline()
        color_cam = pipeline.createColorCamera()
        color_cam.setPreviewSize(self._frame_width, self._frame_height)
        color_cam.setInterleaved(False)
        color_cam.setFps(self._fps)

        xlink_out = pipeline.createXLinkOut()
        xlink_out.setStreamName("color_stream")
        color_cam.preview.link(xlink_out.input)

        if self._device_id:
            all_devices = dai.Device.getAllAvailableDevices()
            matching = [d for d in all_devices if d.getMxId() == self._device_id]
            if not matching:
                raise RuntimeError(f"No OAK device found with serial: {self._device_id!r}")
            self._device = dai.Device(pipeline, matching[0])
        else:
            self._device = dai.Device(pipeline)

        self._out_queue = self._device.getOutputQueue(name="color_stream", maxSize=4, blocking=False)
        self._pipeline = pipeline

    def _capture_loop(self) -> None:
        if self._out_queue is None:
            return
        while self._is_running:
            packet = self._out_queue.tryGet()
            if packet is not None:
                frame_bgr = packet.getCvFrame()
                if frame_bgr is not None:
                    self._set_frame(frame_bgr)
            time.sleep(0.001)


# ============================================
# USB-Webcam (nur Raw-Frames)
# ============================================
class USBWebcamCamera(FrameGrabber):
    """
    FrameGrabber-Implementierung für eine Standard-USB-Webcam via OpenCV.
    """
    def __init__(
        self,
        device_index: int = USB_DEVICE_INDEX,
        width: int = USB_CAPTURE_WIDTH,
        height: int = USB_CAPTURE_HEIGHT,
        fps: int = USB_CAPTURE_FPS
    ) -> None:
        super().__init__()
        self._device_index = device_index
        self._capture = cv2.VideoCapture(self._device_index, cv2.CAP_ANY)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, fps)
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open USB webcam at index {self._device_index}")

    def _capture_loop(self) -> None:
        while self._is_running:
            ret, frame = self._capture.read()
            if not ret:
                time.sleep(0.01)
                continue
            self._set_frame(frame)
            time.sleep(0.001)

    def stop(self) -> None:
        super().stop()
        if self._capture is not None:
            self._capture.release()


# ============================================
# Hilfspaket: Class-wise NMS
# ============================================
def class_wise_nms(boxes: np.ndarray,
                   scores: np.ndarray,
                   classes: np.ndarray,
                   iou_thresh: float) -> list[int]:
    """
    Führt NMS pro Klasse durch.
    boxes: np.ndarray mit Form (N,4) in [x1,y1,x2,y2]
    scores: np.ndarray mit Form (N,)
    classes: np.ndarray mit Form (N,)
    iou_thresh: Schwellenwert für NMS
    Rückgabe: Liste der Indizes, die beibehalten werden.
    """
    keep_indices: list[int] = []
    unique_classes = np.unique(classes)
    for cls in unique_classes:
        inds = np.where(classes == cls)[0]
        cls_boxes = boxes[inds]
        cls_scores = scores[inds]
        order = cls_scores.argsort()[::-1]
        suppressed = np.zeros(len(inds), dtype=bool)
        for i_idx in range(len(order)):
            if suppressed[order[i_idx]]:
                continue
            keep_idx = inds[order[i_idx]]
            keep_indices.append(keep_idx)
            box_i = cls_boxes[order[i_idx]]
            for j_idx in range(i_idx + 1, len(order)):
                if suppressed[order[j_idx]]:
                    continue
                box_j = cls_boxes[order[j_idx]]
                # IoU berechnen
                xx1 = max(box_i[0], box_j[0])
                yy1 = max(box_i[1], box_j[1])
                xx2 = min(box_i[2], box_j[2])
                yy2 = min(box_i[3], box_j[3])
                w = max(0.0, xx2 - xx1)
                h = max(0.0, yy2 - yy1)
                inter = w * h
                area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
                area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
                union = area_i + area_j - inter
                iou = inter / union if union > 0 else 0.0
                if iou > iou_thresh:
                    suppressed[order[j_idx]] = True
    return keep_indices


# ============================================
# DetectCamera mit Ensemble-Prediction
# ============================================
class DetectCamera(FrameGrabber):
    """
    Wrappt einen Roh-FrameGrabber (z.B. USBWebcamCamera) mit zwei YOLO-Engines für Ensemble-Prediction.
    Zusätzlich führt es ArUco-Erkennung auf 320×320 aus (alle ARUCO_SKIP_FRAMES),
    um Pixel→mm-Ratio zu berechnen. Zeigt reale Maße und FPS neben Box-Labels.
    """
    def __init__(
        self,
        base_cam: FrameGrabber,
        engine_path1: str,
        engine_path2: str,
        model_input_size: int = YOLO_MODEL_INPUT_SIZE,
        conf_thresh: float = YOLO_CONF_THRESH,
        iou_thresh: float = YOLO_IOU_THRESH,
        max_det: int = YOLO_MAX_DET,
        device: str = YOLO_DEVICE,
        skip_frames: int = SKIP_FRAMES,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        super().__init__()
        self.base_cam = base_cam
        self.model_input_size = model_input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_det = max_det
        self.device = device
        self.skip_frames = skip_frames
        self.batch_size = batch_size

        # Zwei YOLO-Detect-Engines (TensorRT FP16)
        self.model1 = YOLO(engine_path1, task="detect")
        self.model1.overrides["imgsz"]   = (model_input_size, model_input_size)
        self.model1.overrides["conf"]    = conf_thresh
        self.model1.overrides["iou"]     = iou_thresh
        self.model1.overrides["max_det"] = max_det

        self.model2 = YOLO(engine_path2, task="detect")
        self.model2.overrides["imgsz"]   = (model_input_size, model_input_size)
        self.model2.overrides["conf"]    = conf_thresh
        self.model2.overrides["iou"]     = iou_thresh
        self.model2.overrides["max_det"] = max_det

        self._lock = threading.Lock()
        self._latest_annotated: Optional[np.ndarray] = None
        self._is_running = False
        self._frame_counter = 0
        self.last_known_ratio: Optional[float] = None  # Pixel→mm-Ratio
        self._fps_start_time = time.time()            # Startzeit für FPS-Berechnung
        self._fps_frame_count = 0                     # Frame-Zähler für FPS
        self._fps: float = 0.0                        # zuletzt berechnete FPS

    def start(self) -> None:
        self.base_cam.start()
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._is_running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)

    def _inference_loop(self) -> None:
        while self._is_running:
            raw_full = self.base_cam.get_latest_frame()  # 640×480 BGR
            if raw_full is None:
                time.sleep(0.001)
                continue

            # FPS berechnen (über mehrere Frames hinweg)
            self._fps_frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self._fps_start_time
            
            # FPS alle 30 Frames oder alle 2 Sekunden neu berechnen
            if self._fps_frame_count >= 30 or elapsed_time >= 2.0:
                if elapsed_time > 0:
                    self._fps = self._fps_frame_count / elapsed_time
                self._fps_start_time = current_time
                self._fps_frame_count = 0

            self._frame_counter += 1

            # 1) Downsample auf 320×320 und BGR→RGB (für beide Modelle + ArUco)
            frame_320 = cv2.resize(
                raw_full,
                (self.model_input_size, self.model_input_size),
                interpolation=cv2.INTER_LINEAR,
            )
            img_rgb = cv2.cvtColor(frame_320, cv2.COLOR_BGR2RGB)

            # 2) ArUco nur, wenn enabled UND bestimmte Frame-Nummer:
            if ARUCO_ENABLED and (self._frame_counter % ARUCO_SKIP_FRAMES) == 0:
                ratio, _ = detect_aruco_markers_with_tracking(frame_320)
                if ratio is not None:
                    self.last_known_ratio = ratio

            # 3) Skip-Frames für YOLO
            if (self._frame_counter % self.skip_frames) != 0:
                time.sleep(0.001)
                continue

            # 4) Ensemble-Prediction: beide Modelle gleichzeitig inferieren
            results1 = self.model1.predict(
                source=[img_rgb],
                imgsz=self.model_input_size,
                device=self.device,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                max_det=self.max_det,
                augment=False,
                verbose=False,
            )
            results2 = self.model2.predict(
                source=[img_rgb],
                imgsz=self.model_input_size,
                device=self.device,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                max_det=self.max_det,
                augment=False,
                verbose=False,
            )

            # 5) Extrahiere Boxen, Scores, Klassen beider Ergebnisse
            boxes1   = results1[0].boxes.xyxy.cpu().numpy()
            scores1  = results1[0].boxes.conf.cpu().numpy()
            classes1 = results1[0].boxes.cls.cpu().numpy().astype(int)

            boxes2   = results2[0].boxes.xyxy.cpu().numpy()
            scores2  = results2[0].boxes.conf.cpu().numpy()
            classes2 = results2[0].boxes.cls.cpu().numpy().astype(int)

            if boxes1.size:
                all_boxes   = np.vstack((boxes1, boxes2)) if boxes2.size else boxes1.copy()
                all_scores  = np.hstack((scores1, scores2)) if scores2.size else scores1.copy()
                all_classes = np.hstack((classes1, classes2)) if classes2.size else classes1.copy()
            else:
                all_boxes   = boxes2.copy()
                all_scores  = scores2.copy()
                all_classes = classes2.copy()

            # 6) Gemeinsame NMS über alle Klassen
            if all_boxes.size:
                keep_inds = class_wise_nms(all_boxes, all_scores, all_classes, self.iou_thresh)
                final_boxes   = all_boxes[keep_inds]
                final_scores  = all_scores[keep_inds]
                final_classes = all_classes[keep_inds]
            else:
                final_boxes   = np.zeros((0, 4), dtype=int)
                final_scores  = np.zeros((0,), dtype=float)
                final_classes = np.zeros((0,), dtype=int)

            # 7) Zeichne die kombinierten Boxen
            annotated_320 = frame_320.copy()
            for (x1, y1, x2, y2), conf, cls_id in zip(final_boxes.astype(int), final_scores, final_classes):
                try:
                    class_name = self.model1.names[cls_id]
                except (KeyError, IndexError):
                    class_name = f"class{cls_id}"

                # Reale Maße falls Ratio bekannt
                if self.last_known_ratio is not None:
                    w_mm, h_mm = calculate_dimensions([x1, y1, x2, y2], self.last_known_ratio)
                    if w_mm is not None and h_mm is not None:
                        label = f"{class_name} {conf:.2f}: {w_mm:.1f}x{h_mm:.1f}mm"
                    else:
                        label = f"{class_name} {conf:.2f}"
                else:
                    label = f"{class_name} {conf:.2f}"

                cv2.rectangle(annotated_320, (x1, y1), (x2, y2), (0, 255, 0), 2)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(
                    annotated_320,
                    (x1, y1 - t_size[1] - 4),
                    (x1 + t_size[0], y1),
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    annotated_320,
                    label,
                    (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    thickness=1,
                )

            # 8) FPS und Frame-Nummer einblenden
            fps_text = f"FPS: {self._fps:.1f}"
            cv2.putText(
                annotated_320,
                fps_text,
                (5, self.model_input_size - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                thickness=1,
            )
            frame_text = f"Center Cam Frame #{self._frame_counter}"
            cv2.putText(
                annotated_320,
                frame_text,
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                thickness=1,
            )

            # 9) Speichern
            with self._lock:
                self._latest_annotated = annotated_320

            time.sleep(0.001)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_annotated is None:
                return None
            return self._latest_annotated.copy()


# ============================================
# SegmentCamera (Masken-Inference + FPS)
# ============================================
class SegmentCamera(FrameGrabber):
    """
    Wrappt einen Roh-FrameGrabber (z.B. OAK1MaxCamera) mit YOLO-Engine für Segmentierung.
    Zeigt ebenfalls FPS und Frame-Nummer an.
    """
    def __init__(
        self,
        base_cam: FrameGrabber,
        engine_path: str,
        model_input_size: int = YOLO_MODEL_INPUT_SIZE,
        conf_thresh: float = YOLO_CONF_THRESH,
        iou_thresh: float = YOLO_IOU_THRESH,
        max_det: int = YOLO_MAX_DET,
        device: str = YOLO_DEVICE,
        skip_frames: int = SKIP_FRAMES,
        batch_size: int = BATCH_SIZE,
        rotation: str = "none",  # Neu: Rotation Parameter
    ) -> None:
        super().__init__()
        self.base_cam = base_cam
        self.model_input_size = model_input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_det = max_det
        self.device = device
        self.skip_frames = skip_frames
        self.batch_size = batch_size
        self.rotation = rotation  # Speichere Rotation

        # YOLO-Segment-Engine (TensorRT FP16)
        self.model = YOLO(engine_path, task="segment")
        self.model.overrides["imgsz"]   = (model_input_size, model_input_size)
        self.model.overrides["conf"]    = conf_thresh
        self.model.overrides["iou"]     = iou_thresh
        self.model.overrides["max_det"] = max_det

        self._lock = threading.Lock()
        self._latest_annotated_320: Optional[np.ndarray] = None
        self._is_running = False
        self._frame_counter = 0
        self._fps_start_time = time.time()
        self._fps_frame_count = 0
        self._fps: float = 0.0

    def start(self) -> None:
        self.base_cam.start()
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._is_running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)

    def _inference_loop(self) -> None:
        while self._is_running:
            raw_full = self.base_cam.get_latest_frame()
            if raw_full is None:
                time.sleep(0.001)
                continue

            # FPS berechnen (über mehrere Frames hinweg)
            self._fps_frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self._fps_start_time
            
            # FPS alle 30 Frames oder alle 2 Sekunden neu berechnen
            if self._fps_frame_count >= 30 or elapsed_time >= 2.0:
                if elapsed_time > 0:
                    self._fps = self._fps_frame_count / elapsed_time
                self._fps_start_time = current_time
                self._fps_frame_count = 0

            self._frame_counter += 1
            if (self._frame_counter % self.skip_frames) != 0:
                time.sleep(0.001)
                continue

            # Downsample auf 320×320 und BGR→RGB
            frame_320 = cv2.resize(
                raw_full,
                (self.model_input_size, self.model_input_size),
                interpolation=cv2.INTER_LINEAR,
            )
            img_rgb = cv2.cvtColor(frame_320, cv2.COLOR_BGR2RGB)

            # YOLO Segment Inferenz
            results = self.model.predict(
                source=[img_rgb],
                imgsz=self.model_input_size,
                device=self.device,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                max_det=self.max_det,
                augment=False,
                verbose=False,
            )

            # Masken-Overlay zeichnen
            seg_overlay_rgb = results[0].plot()
            annotated_320 = cv2.cvtColor(seg_overlay_rgb, cv2.COLOR_RGB2BGR)

            # Rotation ZUERST anwenden (vor FPS-Overlay)
            if self.rotation == "left_90":  # Linke Kamera: 90° im Uhrzeigersinn
                annotated_320 = cv2.rotate(annotated_320, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation == "right_90":  # Rechte Kamera: 90° gegen Uhrzeigersinn
                annotated_320 = cv2.rotate(annotated_320, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # NACH der Rotation: FPS-Anzeige
            fps_text = f"FPS: {self._fps:.1f}"
            cv2.putText(
                annotated_320,
                fps_text,
                (5, annotated_320.shape[0] - 5),  # Dynamische Position basierend auf aktueller Höhe
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                thickness=1,
            )

            # Frame-Nummer einblenden
            label = f"Frame #{self._frame_counter}"
            cv2.putText(
                annotated_320,
                label,
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                thickness=1,
            )

            with self._lock:
                self._latest_annotated_320 = annotated_320

            time.sleep(0.001)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_annotated_320 is None:
                return None
            return self._latest_annotated_320.copy()


# ============================================
# Hilfsfunktion: Alle OAK-Seriennummern finden
# ============================================
def list_oak_devices() -> list[str]:
    return [device_info.getMxId() for device_info in dai.Device.getAllAvailableDevices()]


oak_serials = list_oak_devices()
print(f"OAK Serials: {oak_serials}")
if len(oak_serials) < 2:
    raise RuntimeError("Weniger als zwei OAK-1 Max Kameras gefunden. Bitte zwei anschließen.")

left_oak_serial  = oak_serials[0]
right_oak_serial = oak_serials[1]


# ============================================
# Instanziiere alle Kameras & starte Threads
# ============================================
oak_left_raw   = OAK1MaxCamera(device_id=left_oak_serial,  width=OAK_FRAME_WIDTH, height=OAK_FRAME_HEIGHT, fps=OAK_FPS)
oak_left_seg   = SegmentCamera(
    base_cam=oak_left_raw,
    engine_path=YOLO_ENGINE_LEFT,
    model_input_size=YOLO_MODEL_INPUT_SIZE,
    conf_thresh=YOLO_CONF_THRESH,
    iou_thresh=YOLO_IOU_THRESH,
    max_det=YOLO_MAX_DET,
    device=YOLO_DEVICE,
    skip_frames=SKIP_FRAMES,
    batch_size=BATCH_SIZE,
    rotation="left_90",  # Linke Kamera: 90° im Uhrzeigersinn
)

oak_right_raw  = OAK1MaxCamera(device_id=right_oak_serial, width=OAK_FRAME_WIDTH, height=OAK_FRAME_HEIGHT, fps=OAK_FPS)
oak_right_seg  = SegmentCamera(
    base_cam=oak_right_raw,
    engine_path=YOLO_ENGINE_RIGHT,
    model_input_size=YOLO_MODEL_INPUT_SIZE,
    conf_thresh=YOLO_CONF_THRESH,
    iou_thresh=YOLO_IOU_THRESH,
    max_det=YOLO_MAX_DET,
    device=YOLO_DEVICE,
    skip_frames=SKIP_FRAMES,
    batch_size=BATCH_SIZE,
    rotation="right_90",  # Rechte Kamera: 90° gegen Uhrzeigersinn
)

usb_center_raw    = USBWebcamCamera(device_index=USB_DEVICE_INDEX, width=USB_CAPTURE_WIDTH, height=USB_CAPTURE_HEIGHT, fps=USB_CAPTURE_FPS)
usb_center_detect = DetectCamera(
    base_cam=usb_center_raw,
    engine_path1=YOLO_ENGINE_CENTER_1,
    engine_path2=YOLO_ENGINE_CENTER_2,
    model_input_size=YOLO_MODEL_INPUT_SIZE,
    conf_thresh=YOLO_CONF_THRESH,
    iou_thresh=YOLO_IOU_THRESH,
    max_det=YOLO_MAX_DET,
    device=YOLO_DEVICE,
    skip_frames=SKIP_FRAMES,
    batch_size=BATCH_SIZE,
)

oak_left_raw.start()
oak_left_seg.start()
oak_right_raw.start()
oak_right_seg.start()
usb_center_raw.start()
usb_center_detect.start()


# ============================================
# MJPEG-Stream-Generator
# ============================================
def mjpeg_stream_generator(camera: FrameGrabber) -> Generator[bytes, None, None]:
    boundary = MJPEG_BOUNDARY
    while True:
        frame = camera.get_latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        success, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            time.sleep(0.01)
            continue

        data = (
            boundary
            + b"\r\n"
            + b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8")
            + jpg.tobytes()
            + b"\r\n"
        )
        yield data
        time.sleep(0.03)  # ~30 FPS


# ============================================
# FASTAPI-Routen
# ============================================
@app.get("/", response_class=Response)
def index(request: Request) -> Response:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_left")
def video_left() -> StreamingResponse:
    # Linke OAK: Segment (Masken-Overlay + FPS) - Rotation in Kamera-Klasse
    return StreamingResponse(
        mjpeg_stream_generator(oak_left_seg),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


@app.get("/video_center")
def video_center() -> StreamingResponse:
    # USB-Webcam: Ensemble-Detect (Bounding-Boxes + reale Maße + FPS)
    return StreamingResponse(
        mjpeg_stream_generator(usb_center_detect),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


@app.get("/video_right")
def video_right() -> StreamingResponse:
    # Rechte OAK: Segment (Masken-Overlay + FPS) - Rotation in Kamera-Klasse
    return StreamingResponse(
        mjpeg_stream_generator(oak_right_seg),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


@app.on_event("shutdown")
def cleanup_cameras() -> None:
    usb_center_detect.stop()
    oak_left_seg.stop()
    oak_right_seg.stop()
    time.sleep(0.5)


# ============================================
# ENTRYPOINT
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
