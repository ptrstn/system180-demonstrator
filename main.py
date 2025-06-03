# main.py

import threading
import time
from typing import Generator, Optional

import cv2
import depthai as dai
import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
        width: int = 640,
        height: int = 480,
        fps: int = 30,
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
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
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


# ------------------------------------------------------------------------------
# Discover connected OAK devices and assign them to "left" and "right"
# ------------------------------------------------------------------------------

def list_oak_devices() -> list[str]:
    """
    Return a list of serial numbers (MxId) for all connected OAK devices.
    """
    return [device_info.getMxId() for device_info in dai.Device.getAllAvailableDevices()]


oak_serials = list_oak_devices()
if len(oak_serials) < 2:
    raise RuntimeError("Less than two OAK-1 Max cameras were found. Please connect at least two.")

# For example: oak_serials might be ["1944301021ED162F00", "1447B2D71F9A"]
left_oak_serial = oak_serials[0]
right_oak_serial = oak_serials[1]

# Instantiate the two OAK-1 Max cameras (depthai) and the USB webcam
oak_left_camera = OAK1MaxCamera(device_id=left_oak_serial, width=640, height=480, fps=30)
oak_left_camera.start()

oak_right_camera = OAK1MaxCamera(device_id=right_oak_serial, width=640, height=480, fps=30)
oak_right_camera.start()

# If your OBSBOT Meet 2 is not on index 0, change `device_index=0` to the correct index
usb_webcam = USBWebcamCamera(device_index=1, width=1280, height=720, fps=30)
usb_webcam.start()


def mjpeg_stream_generator(camera: FrameGrabber) -> Generator[bytes, None, None]:
    """
    Given a FrameGrabber, yield an MJPEG multipart stream forever.
    """
    boundary = b"--frameboundary"
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


@app.get("/", response_class=Response)
def index(request: Request) -> Response:
    """
    Serves the main HTML page (with three <img> tags pointing to the three streams).
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_left")
def video_left() -> StreamingResponse:
    """
    Streams the left OAK-1 Max camera as MJPEG.
    """
    return StreamingResponse(
        mjpeg_stream_generator(oak_left_camera),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


@app.get("/video_center")
def video_center() -> StreamingResponse:
    """
    Streams the center OBSBOT Meet 2 (USB webcam) as MJPEG.
    """
    return StreamingResponse(
        mjpeg_stream_generator(usb_webcam),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


@app.get("/video_right")
def video_right() -> StreamingResponse:
    """
    Streams the right OAK-1 Max camera as MJPEG.
    """
    return StreamingResponse(
        mjpeg_stream_generator(oak_right_camera),
        media_type="multipart/x-mixed-replace; boundary=frameboundary",
    )


@app.on_event("shutdown")
def cleanup_cameras() -> None:
    """
    Ensure all cameras are stopped cleanly when the server shuts down.
    """
    oak_left_camera.stop()
    oak_right_camera.stop()
    usb_webcam.stop()
