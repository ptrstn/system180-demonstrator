import threading
import time
from typing import Generator, Optional, List

import cv2
import depthai as dai
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ============================================
# CAMERA / INFERENCE CONSTANTS (TUNED DOWN)
# ============================================

# OAK-1 Max (DepthAI) camera settings
OAK_FRAME_WIDTH:  int = 640
OAK_FRAME_HEIGHT: int = 480
OAK_FPS:          int = 30

# USB webcam (OBSBOT Meet 2) settings
USB_DEVICE_INDEX:  int = 0
USB_FRAME_WIDTH:   int = 1280
USB_FRAME_HEIGHT:  int = 720
USB_FPS:           int = 30

# MJPEG boundary
MJPEG_BOUNDARY: bytes = b"--frameboundary"

# Paths to TensorRT engines (all FP16 + 320×320)
YOLO_ENGINE_LEFT:   str = "/home/sys180/models/NubsUpDown.engine"
YOLO_ENGINE_RIGHT:  str = "/home/sys180/models/NubsUpDown.engine"
YOLO_ENGINE_CENTER: str = "/home/sys180/models/system180custommodel_v1.engine"

# INFERENCE “LIGHT” SETTINGS
YOLO_MODEL_INPUT_SIZE: int   = 320   # 320×320 input
YOLO_CONF_THRESH:      float = 0.25
YOLO_IOU_THRESH:       float = 0.45
YOLO_MAX_DET:          int   = 300   # fewer max detections
YOLO_DEVICE:           str   = "cuda"  # run TRT on GPU

# Skip frames (so we only do inference ~6 FPS on 30 FPS capture)
SKIP_FRAMES: int = 5

# Batch size = 1
BATCH_SIZE: int = 1


# ============================================
# FASTAPI SETUP
# ============================================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class FrameGrabber:
    """
    Base class for any camera that continuously grabs frames in a background thread
    and keeps only the latest frame.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._is_running = False

    def start(self) -> None:
        """
        Launch the background thread that keeps grabbing frames.
        """
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Signal the thread to stop capturing.
        """
        self._is_running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)

    def _capture_loop(self) -> None:
        """
        Actual capture loop; must be implemented by subclasses.
        Should set self._latest_frame to a valid BGR numpy array.
        """
        raise NotImplementedError("Subclasses must implement _capture_loop")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Return the most recent frame (BGR numpy array) or None if no frame yet.
        """
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def _set_frame(self, frame: np.ndarray) -> None:
        """
        Store the latest frame (called from within a lock).
        """
        with self._lock:
            self._latest_frame = frame


# ------------------------------------------------------------------------------
# OAK-1 Max Camera (LEFT & RIGHT)
# ------------------------------------------------------------------------------
class OAK1MaxCamera(FrameGrabber):
    """
    FrameGrabber implementation for an OAK-1 Max (DepthAI) camera.
    """

    def __init__(
        self,
        device_id: Optional[str],
        width: int = OAK_FRAME_WIDTH,
        height: int = OAK_FRAME_HEIGHT,
        fps: int = OAK_FPS,
    ) -> None:
        super().__init__()
        self._device_id = device_id
        self._frame_width = width
        self._frame_height = height
        self._fps = fps
        self._pipeline: Optional[dai.Pipeline] = None
        self._device: Optional[dai.Device] = None
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
            time.sleep(0.001)  # yield CPU briefly


# ------------------------------------------------------------------------------
# USB Webcam (CENTER)
# ------------------------------------------------------------------------------
class USBWebcamCamera(FrameGrabber):
    """
    FrameGrabber implementation for a standard USB webcam (using OpenCV).
    """

    def __init__(
        self,
        device_index: int = USB_DEVICE_INDEX,
        width: int = USB_CAPTURE_WIDTH,
        height: int = USB_CAPTURE_HEIGHT,
        fps: int = USB_CAPTURE_FPS,
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


# ------------------------------------------------------------------------------
# INFERENCE WRAPPER (ALL CAMERAS – “LIGHT” MODE)
# ------------------------------------------------------------------------------
class InferenceCamera(FrameGrabber):
    """
    Wraps a raw FrameGrabber with a TensorRT YOLO engine (FP16 + 320×320).
    Operation:
      1) Grabs an input frame from base_cam.
      2) Down‐samples to 320×320.
      3) Skips frames so we infer ~6 FPS (SKIP_FRAMES=5).
      4) Draws bounding boxes on the 320×320 result.
      5) Stores the latest annotated 320×320 frame.
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

        # Load the TensorRT engine expecting 320×320 input (FP16)
        self.model = YOLO(engine_path)
        self.model.overrides["imgsz"]   = (model_input_size, model_input_size)
        self.model.overrides["conf"]    = conf_thresh
        self.model.overrides["iou"]     = iou_thresh
        self.model.overrides["max_det"] = max_det

        self._lock = threading.Lock()
        self._latest_annotated_320: Optional[np.ndarray] = None
        self._is_running = False

        # Frame counter for skipping
        self._frame_counter = 0

    def start(self) -> None:
        # Start base camera thread
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
            raw_frame = self.base_cam.get_latest_frame()
            if raw_frame is None:
                time.sleep(0.001)
                continue

            self._frame_counter += 1
            # Only run inference once every skip_frames
            if (self._frame_counter % self.skip_frames) != 0:
                time.sleep(0.001)
                continue

            # Down-sample to 320×320
            frame_320 = cv2.resize(
                raw_frame,
                (self.model_input_size, self.model_input_size),
                interpolation=cv2.INTER_LINEAR,
            )

            # Run inference on 320×320 (FP16)
            results = self.model.predict(
                source=[frame_320],
                imgsz=self.model_input_size,
                device=self.device,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                max_det=self.max_det,
                augment=False,
                verbose=False,
            )

            # Draw boxes on frame_320
            annotated_320 = frame_320.copy()
            dets = results[0].boxes  # type: ignore[attr-defined]
            if dets is not None and len(dets) > 0:
                xyxy_array  = dets.xyxy.cpu().numpy()
                conf_array  = dets.conf.cpu().numpy()
                classes_arr = dets.cls.cpu().numpy().astype(int)

                for box, c, cls_id in zip(xyxy_array, conf_array, classes_arr):
                    x1, y1, x2, y2 = box.astype(int)
                    try:
                        class_name = self.model.names[cls_id]
                    except (KeyError, IndexError):
                        class_name = f"class{cls_id}"
                    label = f"{class_name} {c:.2f}"

                    cv2.rectangle(annotated_320, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    t_size = cv2.getTextSize(label, 0, fontScale=0.4, thickness=1)[0]
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

            # Store latest annotated 320×320
            with self._lock:
                self._latest_annotated_320 = annotated_320

            time.sleep(0.001)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Return the most recent annotated 320×320 frame,
        or None if inference hasn’t produced anything yet.
        """
        with self._lock:
            if self._latest_annotated_320 is None:
                return None
            return self._latest_annotated_320.copy()


# ------------------------------------------------------------------------------
# Discover connected OAK devices and assign them to left/right
# ------------------------------------------------------------------------------
def list_oak_devices() -> list[str]:
    return [device_info.getMxId() for device_info in dai.Device.getAllAvailableDevices()]


oak_serials = list_oak_devices()
print(f"OAK Serials: {oak_serials}")
if len(oak_serials) < 2:
    raise RuntimeError("Less than two OAK-1 Max cameras were found. Please connect at least two.")

left_oak_serial  = oak_serials[0]
right_oak_serial = oak_serials[1]


# ------------------------------------------------------------------------------
# Instantiate RAW and INFERENCE cameras
# ------------------------------------------------------------------------------
# 1) Raw capture for left & right OAK
oak_left_raw  = OAK1MaxCamera(device_id=left_oak_serial,  width=OAK_FRAME_WIDTH, height=OAK_FRAME_HEIGHT, fps=OAK_FPS)
oak_right_raw = OAK1MaxCamera(device_id=right_oak_serial, width=OAK_FRAME_WIDTH, height=OAK_FRAME_HEIGHT, fps=OAK_FPS)

# 2) Raw capture for center USB
usb_center_raw = USBWebcamCamera(device_index=USB_DEVICE_INDEX, width=USB_CAPTURE_WIDTH, height=USB_CAPTURE_HEIGHT, fps=USB_CAPTURE_FPS)

# 3) Wrap each camera in its own InferenceCamera
oak_left_infer = InferenceCamera(
    base_cam=oak_left_raw,
    engine_path=YOLO_ENGINE_LEFT,
    model_input_size=YOLO_MODEL_INPUT_SIZE,
    conf_thresh=YOLO_CONF_THRESH,
    iou_thresh=YOLO_IOU_THRESH,
    max_det=YOLO_MAX_DET,
    device=YOLO_DEVICE,
    skip_frames=SKIP_FRAMES,
    batch_size=BATCH_SIZE,
)

usb_center_infer = InferenceCamera(
    base_cam=usb_center_raw,
    engine_path=YOLO_ENGINE_CENTER,
    model_input_size=YOLO_MODEL_INPUT_SIZE,
    conf_thresh=YOLO_CONF_THRESH,
    iou_thresh=YOLO_IOU_THRESH,
    max_det=YOLO_MAX_DET,
    device=YOLO_DEVICE,
    skip_frames=SKIP_FRAMES,
    batch_size=BATCH_SIZE,
)

oak_right_infer = InferenceCamera(
    base_cam=oak_right_raw,
    engine_path=YOLO_ENGINE_RIGHT,
    model_input_size=YOLO_MODEL_INPUT_SIZE,
    conf_thresh=YOLO_CONF_THRESH,
    iou_thresh=YOLO_IOU_THRESH,
    max_det=YOLO_MAX_DET,
    device=YOLO_DEVICE,
    skip_frames=SKIP_FRAMES,
    batch_size=BATCH_SIZE,
)

# 4) Start all inference threads (which also start raw capture)
oak_left_infer.start()
usb_center_infer.start()
oak_right_infer.start()


# ------------------------------------------------------------------------------
# MJPEG STREAM GENERATOR (UNCHANGED)
# ------------------------------------------------------------------------------
def mjpeg_stream_generator(camera: FrameGrabber) -> Generator[bytes, None, None]:
    boundary = MJPEG_BOUNDARY
    while True:
        frame = camera.get_latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        success, jpg = cv2.imencode(".jpg", frame)
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
        time.sleep(0.03)  # ~30 FPS throttle


# ------------------------------------------------------------------------------
# FASTAPI ROUTES
# ------------------------------------------------------------------------------
@app.get("/", response_class=Response)
def index(request: Request) -> Response:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_left")
def video_left() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_stream_generator(oak_left_infer),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


@app.get("/video_center")
def video_center() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_stream_generator(usb_center_infer),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


@app.get("/video_right")
def video_right() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_stream_generator(oak_right_infer),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


@app.on_event("shutdown")
def cleanup_cameras() -> None:
    # Stop all inference threads (which then stop base cameras)
    oak_left_infer.stop()
    usb_center_infer.stop()
    oak_right_infer.stop()

    # Stop raw capture as well
    oak_left_raw.stop()
    usb_center_raw.stop()
    oak_right_raw.stop()
