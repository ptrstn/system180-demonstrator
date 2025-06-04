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

from measurement_demobau_with_aruco import calculate_dimensions, \
    detect_aruco_markers_with_tracking

# ============================================
# KAMEREN- / INFERENZ-KONSTANTEN
# ============================================

# OAK-1 Max (DepthAI) Kameraeinstellungen (LEFT & RIGHT segmentieren)
OAK_FRAME_WIDTH:           int   = 640
OAK_FRAME_HEIGHT:          int   = 480
OAK_FPS:                   int   = 30
YOLO_ENGINE_LEFT:          str   = "/home/sys180/models/NubsUpDown_320_FP16_segment.engine"
YOLO_ENGINE_RIGHT:         str   = "/home/sys180/models/NubsUpDown_320_FP16_segment.engine"

# USB-Webcam (CENTER detektieren)
USB_DEVICE_INDEX:          int   = 0
USB_CAPTURE_WIDTH:         int   = 640
USB_CAPTURE_HEIGHT:        int   = 480
USB_CAPTURE_FPS:           int   = 30
YOLO_ENGINE_CENTER:        str   = "/home/sys180/models/custom_320_FP16_detect.engine"

# YOLO-Inference-Grundeinstellungen
YOLO_MODEL_INPUT_SIZE:     int   = 320   # <-- Wichtig: neu definiert
YOLO_CONF_THRESH:          float = 0.25
YOLO_IOU_THRESH:           float = 0.45
YOLO_MAX_DET:              int   = 300
YOLO_DEVICE:               str   = "cuda"

# MJPEG-Boundary
MJPEG_BOUNDARY:            bytes = b"--frameboundary"

# Inferenz-Frequenz (skip N-1 Frames)
SKIP_FRAMES:               int   = 5

# Batch-Größe
BATCH_SIZE:                int   = 1


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
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._is_running = False

    def start(self) -> None:
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._is_running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)

    def _capture_loop(self) -> None:
        raise NotImplementedError("Subclasses must implement _capture_loop")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def _set_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._latest_frame = frame


# ============================================
# OAK-1 Max Camera (nur Raw-Frames)
# ============================================
class OAK1MaxCamera(FrameGrabber):
    def __init__(self, device_id: Optional[str], width: int = OAK_FRAME_WIDTH, height: int = OAK_FRAME_HEIGHT, fps: int = OAK_FPS) -> None:
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
    def __init__(self, device_index: int = USB_DEVICE_INDEX, width: int = USB_CAPTURE_WIDTH, height: int = USB_CAPTURE_HEIGHT, fps: int = USB_CAPTURE_FPS) -> None:
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
# DetectCamera (Bounding-Box-Inference)
# ============================================

class DetectCamera(FrameGrabber):
    """
    Wraps a raw USB frame grabber with YOLO-detect (TensorRT FP16, imgsz=320).
    Also runs ArUco to compute pixel→mm ratio.
    """

    def __init__(self, base_cam: FrameGrabber, engine_path: str, ...):
        super().__init__()
        self.base_cam = base_cam
        self.model = YOLO(engine_path)
        self.model.overrides["imgsz"] = (YOLO_MODEL_INPUT_SIZE, YOLO_MODEL_INPUT_SIZE)
        ...
        self.last_known_ratio = None  # store pixel/mm ratio
        self._frame_counter = 0
        self._latest_annotated = None
        self._lock = threading.Lock()

    def start(self):
        self.base_cam.start()
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._is_running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)

    def _inference_loop(self):
        while self._is_running:
            raw = self.base_cam.get_latest_frame()  # full-res BGR
            if raw is None:
                time.sleep(0.001)
                continue

            # 1) ArUco detection once per frame (full resolution)
            ratio, _ = detect_aruco_markers_with_tracking(raw)
            if ratio is not None:
                self.last_known_ratio = ratio

            # 2) Skip frames if desired
            self._frame_counter += 1
            if (self._frame_counter % SKIP_FRAMES) != 0:
                time.sleep(0.001)
                continue

            # 3) Downsample to 320×320
            frame320 = cv2.resize(raw,
                                  (YOLO_MODEL_INPUT_SIZE, YOLO_MODEL_INPUT_SIZE),
                                  interpolation=cv2.INTER_LINEAR)

            # 4) Run YOLO-detect
            results = self.model.predict(
                source=[frame320],
                imgsz=YOLO_MODEL_INPUT_SIZE,
                device=YOLO_DEVICE,
                conf=YOLO_CONF_THRESH,
                iou=YOLO_IOU_THRESH,
                max_det=YOLO_MAX_DET,
                augment=False,
                verbose=False,
            )

            # 5) Draw detections on 320×320 crop, including real-world dims
            annotated = frame320.copy()
            dets = results[0].boxes  # type: ignore[attr-defined]
            if dets is not None and len(dets) > 0:
                xyxy = dets.xyxy.cpu().numpy().astype(int)
                confs = dets.conf.cpu().numpy()
                classes = dets.cls.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, classes):
                    # lookup name
                    try:
                        cls_name = self.model.names[cls_id]
                    except (KeyError, IndexError):
                        cls_name = f"class{cls_id}"

                    # compute mm dims if we have a ratio
                    if self.last_known_ratio is not None:
                        w_mm, h_mm = calculate_dimensions([x1, y1, x2, y2], self.last_known_ratio)
                        label = f"{cls_name} {conf:.2f}: {w_mm:.1f}mm×{h_mm:.1f}mm"
                    else:
                        label = f"{cls_name} {conf:.2f}"

                    # draw box + label
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    tsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(annotated,
                                  (x1, y1 - tsz[1] - 4),
                                  (x1 + tsz[0], y1),
                                  (0, 255, 0), -1)
                    cv2.putText(annotated, label, (x1, y1 - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            # 6) Store latest annotated frame
            with self._lock:
                self._latest_annotated = annotated

            time.sleep(0.001)

    def get_latest_frame(self):
        with self._lock:
            return None if self._latest_annotated is None else self._latest_annotated.copy()


# ============================================
# SegmentCamera (Masken-Inference)
# ============================================
class SegmentCamera(FrameGrabber):
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

        # YOLO-Segment-Engine
        self.model = YOLO(engine_path, task="segment")
        self.model.overrides["imgsz"]   = (model_input_size, model_input_size)
        self.model.overrides["conf"]    = conf_thresh
        self.model.overrides["iou"]     = iou_thresh
        self.model.overrides["max_det"] = max_det

        self._lock = threading.Lock()
        self._latest_annotated_320: Optional[np.ndarray] = None
        self._is_running = False
        self._frame_counter = 0

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

            self._frame_counter += 1
            if (self._frame_counter % self.skip_frames) != 0:
                time.sleep(0.001)
                continue

            frame_320 = cv2.resize(
                raw_full,
                (self.model_input_size, self.model_input_size),
                interpolation=cv2.INTER_LINEAR,
            )
            img_rgb = cv2.cvtColor(frame_320, cv2.COLOR_BGR2RGB)

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

            seg_overlay_rgb = results[0].plot()
            annotated_320 = cv2.cvtColor(seg_overlay_rgb, cv2.COLOR_RGB2BGR)

            # ------------------------------------
            # Frame-Nummer für Left/Right-Kameras einblenden
            # ------------------------------------
            # Hier nehme ich an, dass engine_path schon beides unterscheidet.
            # Empfehlenswert: beim Erzeugen von SegmentCamera einen camera_name übergeben.
            label = f"Frame #{self._frame_counter}"
            # Wenn Sie spezifisch „Left“ oder „Right“ anzeigen wollen, können Sie
            # beim Instanziieren z. B. segment_cam_left = SegmentCamera(..., camera_name="Left")
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
# Alle OAK-Seriennummern auslesen
# ============================================
def list_oak_devices() -> list[str]:
    return [device_info.getMxId() for device_info in dai.Device.getAllAvailableDevices()]


oak_serials = list_oak_devices()
print(f"OAK Serials: {oak_serials}")
if len(oak_serials) < 2:
    raise RuntimeError("Weniger als zwei OAK-1 Max Kameras gefunden. Bitte mindestens zwei anschließen.")

left_oak_serial  = oak_serials[0]
right_oak_serial = oak_serials[1]


# ============================================
# Kameras instanziieren & starten
# ============================================
oak_left_raw   = OAK1MaxCamera(device_id=left_oak_serial,  width=OAK_FRAME_WIDTH,  height=OAK_FRAME_HEIGHT, fps=OAK_FPS)
oak_left_seg   = SegmentCamera(
    base_cam=oak_left_raw,
    engine_path=YOLO_ENGINE_LEFT,
    model_input_size=YOLO_MODEL_INPUT_SIZE,
    conf_thresh=YOLO_CONF_THRESH,
    iou_thresh=YOLO_IOU_THRESH,
    max_det=YOLO_MAX_DET,
    device=YOLO_DEVICE,
    skip_frames=SKIP_FRAMES,
    batch_size=BATCH_SIZE
)

oak_right_raw  = OAK1MaxCamera(device_id=right_oak_serial, width=OAK_FRAME_WIDTH,  height=OAK_FRAME_HEIGHT, fps=OAK_FPS)
oak_right_seg  = SegmentCamera(
    base_cam=oak_right_raw,
    engine_path=YOLO_ENGINE_RIGHT,
    model_input_size=YOLO_MODEL_INPUT_SIZE,
    conf_thresh=YOLO_CONF_THRESH,
    iou_thresh=YOLO_IOU_THRESH,
    max_det=YOLO_MAX_DET,
    device=YOLO_DEVICE,
    skip_frames=SKIP_FRAMES,
    batch_size=BATCH_SIZE
)

usb_center_raw    = USBWebcamCamera(device_index=USB_DEVICE_INDEX, width=USB_CAPTURE_WIDTH, height=USB_CAPTURE_HEIGHT, fps=USB_CAPTURE_FPS)
usb_center_detect = DetectCamera(
    base_cam=usb_center_raw,
    engine_path=YOLO_ENGINE_CENTER,
    model_input_size=YOLO_MODEL_INPUT_SIZE,
    conf_thresh=YOLO_CONF_THRESH,
    iou_thresh=YOLO_IOU_THRESH,
    max_det=YOLO_MAX_DET,
    device=YOLO_DEVICE,
    skip_frames=SKIP_FRAMES,
    batch_size=BATCH_SIZE
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
    # Linke OAK: Masken-Overlay
    return StreamingResponse(
        mjpeg_stream_generator(oak_left_seg),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


@app.get("/video_center")
def video_center() -> StreamingResponse:
    # USB-Webcam: Bounding-Box
    return StreamingResponse(
        mjpeg_stream_generator(usb_center_detect),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


@app.get("/video_right")
def video_right() -> StreamingResponse:
    # Rechte OAK: Masken-Overlay
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
