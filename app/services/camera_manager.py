import asyncio
import cv2
import logging
import time
from typing import Dict, Any, AsyncGenerator, Optional
import numpy as np
import depthai as dai
from app.config import settings
from app.utils.exceptions import CameraError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Camera:
    """Base camera class"""
    def __init__(self, name: str):
        self.name = name
        self.is_connected = False
        self.last_frame = None
        self.last_frame_time = 0
        self.error_message: Optional[str] = None

    def get_status(self) -> Dict[str, Any]:
        """Get camera status information"""
        return {
            "name": self.name,
            "connected": self.is_connected,
            "fps": self.get_current_fps() if self.is_connected else 0,
            "error": self.error_message
        }
    
    def get_current_fps(self) -> float:
        """Calculate approximate FPS based on frame timestamps"""
        if self.last_frame_time == 0:
            return 0
        
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        
        if time_diff > 0:
            return 1.0 / time_diff
        return 0

class OakCamera(Camera):
    """OAK-1 MAX camera implementation"""
    def __init__(self, name: str, mx_id: Optional[str] = None):
        super().__init__(name)
        self.mx_id = mx_id
        self.device = None
        self.pipeline = None
        self.camera_rgb = None
        self.resolution = settings.OAK_LEFT_RESOLUTION if "left" in name else settings.OAK_RIGHT_RESOLUTION
        self.fps = settings.OAK_LEFT_FPS if "left" in name else settings.OAK_RIGHT_FPS
        self.queue = None
    
    async def initialize(self) -> bool:
        """Initialize the OAK camera"""
        try:
            # Create pipeline
            self.pipeline = dai.Pipeline()
            
            # Define sources and outputs
            self.camera_rgb = self.pipeline.create(dai.node.ColorCamera)
            xout_rgb = self.pipeline.create(dai.node.XLinkOut)
            
            xout_rgb.setStreamName("rgb")
            
            # Properties
            self.camera_rgb.setPreviewSize(self.resolution[0], self.resolution[1])
            self.camera_rgb.setInterleaved(False)
            self.camera_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            self.camera_rgb.setFps(self.fps)
            
            # Linking
            self.camera_rgb.preview.link(xout_rgb.input)
            
            # Connect to device
            device_info = dai.DeviceInfo()
            if self.mx_id:
                device_info.mxId = self.mx_id
            
            self.device = dai.Device(self.pipeline, device_info)
            self.queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            
            self.is_connected = True
            self.error_message = None
            logger.info(f"OAK camera {self.name} initialized successfully")
            return True
        
        except Exception as e:
            self.error_message = str(e)
            logger.error(f"Failed to initialize OAK camera {self.name}: {e}")
            self.is_connected = False
            return False
    
    async def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame from the camera"""
        if not self.is_connected or self.queue is None:
            return None
        
        try:
            in_rgb = self.queue.get()
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                self.last_frame = frame
                self.last_frame_time = time.time()
                return frame
        except Exception as e:
            logger.error(f"Error getting frame from OAK camera {self.name}: {e}")
        
        return self.last_frame
    
    async def close(self) -> None:
        """Close the camera and release resources"""
        if self.device is not None:
            try:
                self.device.close()
                self.is_connected = False
                logger.info(f"OAK camera {self.name} closed")
            except Exception as e:
                logger.error(f"Error closing OAK camera {self.name}: {e}")

class StandardCamera(Camera):
    """Standard webcam implementation using OpenCV"""
    def __init__(self, name: str, device_id: int = 0):
        super().__init__(name)
        self.device_id = device_id
        self.cap = None
        self.resolution = settings.OBSBOT_RESOLUTION
        self.fps = settings.OBSBOT_FPS
    
    async def initialize(self) -> bool:
        """Initialize the standard webcam"""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                self.error_message = f"Could not open camera with device ID {self.device_id}"
                logger.error(self.error_message)
                return False
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Set FPS
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Read initial frame to verify camera works
            ret, _ = self.cap.read()
            if not ret:
                self.error_message = "Could not read frame from camera"
                logger.error(self.error_message)
                self.cap.release()
                self.cap = None
                return False
            
            self.is_connected = True
            self.error_message = None
            logger.info(f"Standard camera {self.name} initialized successfully")
            return True
        
        except Exception as e:
            self.error_message = str(e)
            logger.error(f"Failed to initialize standard camera {self.name}: {e}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False
    
    async def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame from the camera"""
        if not self.is_connected or self.cap is None:
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame
                self.last_frame_time = time.time()
                return frame
        except Exception as e:
            logger.error(f"Error getting frame from standard camera {self.name}: {e}")
        
        return self.last_frame
    
    async def close(self) -> None:
        """Close the camera and release resources"""
        if self.cap is not None:
            try:
                self.cap.release()
                self.is_connected = False
                logger.info(f"Standard camera {self.name} closed")
            except Exception as e:
                logger.error(f"Error closing standard camera {self.name}: {e}")

class CameraManager:
    """Singleton manager for all cameras"""
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = CameraManager()
        return cls._instance
    
    def __init__(self):
        if CameraManager._instance is not None:
            raise RuntimeError("Use get_instance() to get the singleton instance")
        
        self.oak_left = OakCamera("OAK-1 MAX (Left)", settings.get_oak_left_config().mx_id)
        self.obsbot = StandardCamera("OBSBOT Meet 2 4K", settings.get_obsbot_config().device_id)
        self.oak_right = OakCamera("OAK-1 MAX (Right)", settings.get_oak_right_config().mx_id)
        
        self.encoding_parameters = [
            cv2.IMWRITE_JPEG_QUALITY, 
            settings.STREAM_QUALITY
        ]
    
    async def initialize_cameras(self) -> Dict[str, bool]:
        """Initialize all cameras asynchronously"""
        tasks = [
            self.oak_left.initialize(),
            self.obsbot.initialize(),
            self.oak_right.initialize()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "oak_left": isinstance(results[0], bool) and results[0],
            "obsbot": isinstance(results[1], bool) and results[1],
            "oak_right": isinstance(results[2], bool) and results[2]
        }
    
    def get_camera_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all cameras"""
        return {
            "oak_left": self.oak_left.get_status(),
            "obsbot": self.obsbot.get_status(),
            "oak_right": self.oak_right.get_status()
        }
    
    async def close_all_cameras(self) -> None:
        """Close all cameras"""
        await asyncio.gather(
            self.oak_left.close(),
            self.obsbot.close(),
            self.oak_right.close()
        )
    
    async def _generate_frames(self, camera: Camera) -> AsyncGenerator[bytes, None]:
        """Generate MJPEG frames for streaming"""
        while True:
            try:
                frame = await camera.get_frame()
                
                if frame is None:
                    # If no frame is available, wait and try again
                    await asyncio.sleep(0.1)
                    continue
                
                # Encode frame to JPEG
                success, encoded_frame = cv2.imencode(
                    '.jpg', 
                    frame, 
                    self.encoding_parameters
                )
                
                if not success:
                    logger.error(f"Failed to encode frame from {camera.name}")
                    await asyncio.sleep(0.1)
                    continue
                
                # Yield the frame in MJPEG format
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + 
                    encoded_frame.tobytes() + 
                    b'\r\n'
                )
                
                # Short sleep to control frame rate
                await asyncio.sleep(1/30)  # Limit to ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in frame generation for {camera.name}: {e}")
                await asyncio.sleep(0.5)  # Longer sleep on error
    
    def get_oak_left_stream(self) -> AsyncGenerator[bytes, None]:
        """Get stream for left OAK camera"""
        if not self.oak_left.is_connected:
            raise CameraError("Left OAK camera is not connected")
        return self._generate_frames(self.oak_left)
    
    def get_obsbot_stream(self) -> AsyncGenerator[bytes, None]:
        """Get stream for OBSBOT camera"""
        if not self.obsbot.is_connected:
            raise CameraError("OBSBOT camera is not connected")
        return self._generate_frames(self.obsbot)
    
    def get_oak_right_stream(self) -> AsyncGenerator[bytes, None]:
        """Get stream for right OAK camera"""
        if not self.oak_right.is_connected:
            raise CameraError("Right OAK camera is not connected")
        return self._generate_frames(self.oak_right)