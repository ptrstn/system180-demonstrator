import threading
import time
from typing import Generator, Optional, List

import cv2
import depthai as dai
import numpy as np
import torch
from ultralytics import YOLO
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------
# OAK-1 Max (DepthAI) camera settings
OAK_FRAME_WIDTH: int = 640
OAK_FRAME_HEIGHT: int = 480
OAK_FPS: int = 30

# USB webcam (OBSBOT Meet 2) settings
USB_DEVICE_INDEX: int = 1
USB_FRAME_WIDTH: int = 1280
USB_FRAME_HEIGHT: int = 720
USB_FPS: int = 30

# MJPEG streaming boundary
MJPEG_BOUNDARY: bytes = b"--frameboundary"

# YOLOv11 model paths (one per camera)
YOLO_MODEL_LEFT: str = "yolov11_cam1.pt"
YOLO_MODEL_CENTER: str = "yolov11_cam2.engine"
YOLO_MODEL_RIGHT: str = "yolov11_cam3.pt"

# Inference settings (common defaults)
YOLO_IMG_SIZE: int = 640
YOLO_CONF_THRESH: float = 0.25
YOLO_IOU_THRESH: float = 0.45
YOLO_MAX_DET: int = 1000
YOLO_DEVICE: str = "cuda"  # on Jetson Nano Orin, use GPU


# --------------------------------------------------------------------------
# FASTAPI SETUP
# --------------------------------------------------------------------------
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
            # Return a copy to avoid thread‐safety issues downstream
            return self._latest_frame.copy()

    def _set_frame(self, frame: np.ndarray) -> None:
        """
        Store the latest frame (called from within a lock).
        """
        with self._lock:
            self._latest_frame = frame


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
        """
        :param device_id: The OAK device serial number string, or None to pick the first available.
        :param width: Desired color frame width.
        :param height: Desired color frame height.
        :param fps: Desired color frame rate.
        """
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
        """
        Build and start the DepthAI pipeline for color frames only.
        """
        pipeline = dai.Pipeline()

        # Create a ColorCamera node
        color_cam = pipeline.createColorCamera()
        color_cam.setPreviewSize(self._frame_width, self._frame_height)
        color_cam.setInterleaved(False)
        color_cam.setFps(self._fps)

        # Create an XLinkOut so we can receive preview frames on the host
        xlink_out = pipeline.createXLinkOut()
        xlink_out.setStreamName("color_stream")
        color_cam.preview.link(xlink_out.input)

        # Choose which device to open. If a serial is provided, find its DeviceInfo first.
        if self._device_id:
            # Look through all connected OAKs for a matching serial
            all_devices_info = dai.Device.getAllAvailableDevices()
            matching_devices = [
                info for info in all_devices_info if info.getMxId() == self._device_id
            ]
            if not matching_devices:
                raise RuntimeError(f"No OAK device found with serial: {self._device_id!r}")
            # Use the first matching DeviceInfo
            chosen_info = matching_devices[0]
            self._device = dai.Device(pipeline, chosen_info)
        else:
            # If no device_id is given, just grab the first OAK on USB
            self._device = dai.Device(pipeline)

        self._out_queue = self._device.getOutputQueue(
            name="color_stream", maxSize=4, blocking=False
        )
        self._pipeline = pipeline

    def _capture_loop(self) -> None:
        """
        Continuously read frames from the depthai queue and store them.
        """
        if self._out_queue is None:
            return

        while self._is_running:
            in_packet = self._out_queue.tryGet()
            if in_packet is not None:
                # depthai frames come as ImgFrame or VideoFrame; getCvFrame() → BGR numpy
                frame_bgr = in_packet.getCvFrame()
                if frame_bgr is not None:
                    self._set_frame(frame_bgr)
            # If no packet is ready, yield CPU briefly
            time.sleep(0.001)


class USBWebcamCamera(FrameGrabber):
    """
    FrameGrabber implementation for a standard USB webcam (using OpenCV).
    """

    def __init__(
        self,
        device_index: int = USB_DEVICE_INDEX,
        width: int = USB_FRAME_WIDTH,
        height: int = USB_FRAME_HEIGHT,
        fps: int = USB_FPS,
    ) -> None:
        """
        :param device_index: OpenCV device index (e.g. 0, 1, 2…).
        :param width: Desired capture width.
        :param height: Desired capture height.
        :param fps: Desired capture fps.
        """
        super().__init__()
        self._device_index = device_index
        self._capture = cv2.VideoCapture(self._device_index, cv2.CAP_ANY)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, fps)

        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open USB webcam at index {self._device_index}")

    def _capture_loop(self) -> None:
        """
        Continuously read frames from the OpenCV VideoCapture device.
        """
        while self._is_running:
            ret, frame = self._capture.read()
            if not ret:
                # If frame read fails, wait and retry
                time.sleep(0.01)
                continue
            self._set_frame(frame)
            time.sleep(0.001)

    def stop(self) -> None:
        """
        Stop capturing and release the VideoCapture.
        """
        super().stop()
        if self._capture is not None:
            self._capture.release()


# ----------------------------------------------------------------------------------
# INFERENCE WRAPPER: Runs YOLOv11 on GPU in a separate thread, draws boxes, and stores
# the annotated frame in place of the raw frame. Streams exactly like FrameGrabber.
# ----------------------------------------------------------------------------------

class InferenceCamera(FrameGrabber):
    """
    Wraps an existing FrameGrabber (raw video source) with a YOLOv11 model.
    This class runs a separate inference thread that:
      1. Pulls the latest raw frame from `base_cam`
      2. Runs YOLOv11 inference on GPU (half precision)
      3. Draws the bounding boxes & labels on the frame
      4. Stores the annotated frame for streaming
    """

    def __init__(
        self,
        base_cam: FrameGrabber,
        model_path: str,
        device: str = YOLO_DEVICE,
        img_size: int = YOLO_IMG_SIZE,
        conf_thresh: float = YOLO_CONF_THRESH,
        iou_thresh: float = YOLO_IOU_THRESH,
        agnostic_nms: bool = False,
        classes: Optional[List[int]] = None,
        max_det: int = YOLO_MAX_DET,
    ) -> None:
        """
        :param base_cam:              The raw FrameGrabber (e.g. OAK1MaxCamera or USBWebcamCamera).
        :param model_path:            Path to your YOLOv11 .pt or .engine file.
        :param device:                "cuda" (on Orin) or "cpu".
        :param img_size:              YOLO’s inference size (images will be letterboxed to this square).
        :param conf_thresh:           Confidence threshold for detections.
        :param iou_thresh:            IoU threshold for NMS.
        :param agnostic_nms:          Whether to run class-agnostic NMS.
        :param classes:               If you only want certain class IDs, supply a list here.
        :param max_det:               Max detections per image (usually 300–1000).
        """
        super().__init__()
        self.base_cam = base_cam
        self.device = device
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.agnostic_nms = agnostic_nms
        self.classes = classes
        self.max_det = max_det

        # Load the YOLOv11 model:
        # If you have a .engine (TensorRT) file, you can pass it here directly
        # (e.g. model = YOLO("yolov11_camX.engine")).
        self.model = YOLO(model_path)
        self.model.to(self.device)             # move to GPU
        self.model.overrides["imgsz"] = (img_size, img_size)
        self.model.overrides["conf"] = conf_thresh
        self.model.overrides["iou"] = iou_thresh
        self.model.overrides["max_det"] = max_det
        if classes is not None:
            self.model.overrides["classes"] = classes

        # Force FP16 on Jetson Orin (TensorRT path will also run in FP16 by default if supported)
        try:
            self.model.model.half()
        except Exception:
            # Some exported engines don’t accept .half(); skip if that fails.
            pass

        self._lock = threading.Lock()
        self._latest_annotated: Optional[np.ndarray] = None
        self._is_running = False

        # (Optional) Pre-warm the model on a dummy frame to avoid cold-start lag:
        # dummy = np.zeros((OAK_FRAME_HEIGHT, OAK_FRAME_WIDTH, 3), dtype=np.uint8)
        # with torch.no_grad():
        #     _ = self.model.predict(source=[dummy], imgsz=self.img_size, device=self.device)

    def start(self) -> None:
        """
        1) Start the base camera’s capture thread (if not already running).
        2) Launch our inference thread.
        """
        # Ensure the base camera is already running
        self.base_cam.start()

        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Stop inference; leave the base camera to be closed by the caller if desired.
        """
        self._is_running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)

    def _inference_loop(self) -> None:
        """
        Continuously:
          1. Grab the most recent raw frame from base_cam
          2. Run YOLOv11 inference on it (on GPU)
          3. Draw boxes/labels on the frame
          4. Store annotated frame into self._latest_annotated
        """
        while self._is_running:
            raw = self.base_cam.get_latest_frame()
            if raw is None:
                # No raw frame yet; wait a millisecond
                time.sleep(0.001)
                continue

            # 1) Pre-process & run inference
            with torch.no_grad():
                results = self.model.predict(
                    source=[raw],                # single‐image batch
                    imgsz=self.img_size,
                    device=self.device,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    max_det=self.max_det,
                    agnostic_nms=self.agnostic_nms,
                    classes=self.classes,
                    augment=False,
                    verbose=False,
                )

            # 2) Extract boxes, confidences, class IDs, and labels
            annotated = raw.copy()
            dets = results[0].boxes  # type: ignore[attr-defined]
            if dets is not None and len(dets) > 0:
                xyxy = dets.xyxy.cpu().numpy()
                conf = dets.conf.cpu().numpy()
                classes_arr = dets.cls.cpu().numpy().astype(int)

                for box, c, cls_id in zip(xyxy, conf, classes_arr):
                    x1, y1, x2, y2 = box.astype(int)
                    label = f"{self.model.names[cls_id]} {c:.2f}"
                    # Draw rectangle + filled label background
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                    cv2.rectangle(
                        annotated,
                        (x1, y1 - t_size[1] - 4),
                        (x1 + t_size[0], y1),
                        (0, 255, 0),
                        -1,
                    )
                    cv2.putText(
                        annotated,
                        label,
                        (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        thickness=1,
                    )

            # 3) Store annotated frame
            with self._lock:
                self._latest_annotated = annotated

            # Throttle slightly to avoid burning 100% CPU
            time.sleep(0.001)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Return the most recent annotated frame (BGR numpy array), or None if
        inference hasn’t produced anything yet.
        """
        with self._lock:
            if self._latest_annotated is None:
                return None
            return self._latest_annotated.copy()


# ------------------------------------------------------------------------------
# DISCOVER connected OAK devices and assign them to "left" and "right"
# ------------------------------------------------------------------------------

def list_oak_devices() -> list[str]:
    """
    Return a list of serial numbers (MxId) for all connected OAK devices.
    """
    return [device_info.getMxId() for device_info in dai.Device.getAllAvailableDevices()]


oak_serials = list_oak_devices()
print(f"OAK Serials: {oak_serials}")
if len(oak_serials) < 2:
    raise RuntimeError("Less than two OAK-1 Max cameras were found. Please connect at least two.")

left_oak_serial = oak_serials[0]
right_oak_serial = oak_serials[1]


# --------------------------------------------------------------------------
# Instantiate RAW and INFERENCE cameras
# --------------------------------------------------------------------------
# 1) Raw capture objects (unchanged)
oak_left_raw   = OAK1MaxCamera(device_id=left_oak_serial,  width=OAK_FRAME_WIDTH, height=OAK_FRAME_HEIGHT, fps=OAK_FPS)
oak_right_raw  = OAK1MaxCamera(device_id=right_oak_serial, width=OAK_FRAME_WIDTH, height=OAK_FRAME_HEIGHT, fps=OAK_FPS)
usb_center_raw = USBWebcamCamera(device_index=USB_DEVICE_INDEX, width=USB_FRAME_WIDTH, height=USB_FRAME_HEIGHT, fps=USB_FPS)

# 2) Wrap each raw grabber in its own InferenceCamera, using the constants:
oak_left_infer   = InferenceCamera(
    base_cam=oak_left_raw,
    model_path=YOLO_MODEL_LEFT,
    device=YOLO_DEVICE,
    img_size=YOLO_IMG_SIZE,
    conf_thresh=YOLO_CONF_THRESH,
    iou_thresh=YOLO_IOU_THRESH,
    max_det=YOLO_MAX_DET,
)
oak_right_infer  = InferenceCamera(
    base_cam=oak_right_raw,
    model_path=YOLO_MODEL_RIGHT,
    device=YOLO_DEVICE,
    img_size=YOLO_IMG_SIZE,
    conf_thresh=YOLO_CONF_THRESH,
    iou_thresh=YOLO_IOU_THRESH,
    max_det=YOLO_MAX_DET,
)
usb_center_infer = InferenceCamera(
    base_cam=usb_center_raw,
    model_path=YOLO_MODEL_CENTER,
    device=YOLO_DEVICE,
    img_size=YOLO_IMG_SIZE,
    conf_thresh=YOLO_CONF_THRESH,
    iou_thresh=YOLO_IOU_THRESH,
    max_det=YOLO_MAX_DET,
)

# 3) Start both raw and inference threads
oak_left_infer.start()
oak_right_infer.start()
usb_center_infer.start()


# ------------------------------------------------------------------------------
# MJPEG STREAM GENERATOR (unchanged)
# ------------------------------------------------------------------------------

def mjpeg_stream_generator(camera: FrameGrabber) -> Generator[bytes, None, None]:
    """
    Given a FrameGrabber, yield an MJPEG multipart stream forever.
    """
    boundary = MJPEG_BOUNDARY
    while True:
        frame = camera.get_latest_frame()
        if frame is None:
            # No frame ready yet; wait a bit
            time.sleep(0.01)
            continue

        # Encode BGR frame to JPEG
        success, encoded_image = cv2.imencode(".jpg", frame)
        if not success:
            time.sleep(0.01)
            continue

        jpeg_bytes = encoded_image.tobytes()

        # Build multipart MJPEG chunk
        yield (
            boundary
            + b"\r\n"
            + b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode("utf-8")
            + jpeg_bytes
            + b"\r\n"
        )
        # Optionally throttle to roughly ~30 fps
        time.sleep(0.03)


# ------------------------------------------------------------------------------
# FASTAPI ROUTES (unchanged except for using the inference wrappers)
# ------------------------------------------------------------------------------

@app.get("/", response_class=Response)
def index(request: Request) -> Response:
    """
    Serves the main HTML page (with three <img> tags pointing to the three streams).
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_left")
def video_left() -> StreamingResponse:
    """
    Streams the left OAK-1 Max camera (with YOLO inference) as MJPEG.
    """
    return StreamingResponse(
        mjpeg_stream_generator(oak_left_infer),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


@app.get("/video_center")
def video_center() -> StreamingResponse:
    """
    Streams the center USB webcam camera (with YOLO inference) as MJPEG.
    """
    return StreamingResponse(
        mjpeg_stream_generator(usb_center_infer),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


@app.get("/video_right")
def video_right() -> StreamingResponse:
    """
    Streams the right OAK-1 Max camera (with YOLO inference) as MJPEG.
    """
    return StreamingResponse(
        mjpeg_stream_generator(oak_right_infer),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


@app.on_event("shutdown")
def cleanup_cameras() -> None:
    """
    Ensure all cameras (raw + inference) are stopped cleanly when the server shuts down.
    """
    oak_left_infer.stop()
    oak_right_infer.stop()
    usb_center_infer.stop()

    # If you also want to explicitly stop the raw grabbers:
    oak_left_raw.stop()
    oak_right_raw.stop()
    usb_center_raw.stop()
