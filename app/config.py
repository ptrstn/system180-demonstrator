from pydantic_settings import BaseSettings
from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class CameraConfig:
    """Base class for camera configuration"""
    def __init__(self, resolution: tuple[int, int], fps: int):
        self.resolution = resolution
        self.fps = fps

class OakCameraConfig(CameraConfig):
    """Configuration for OAK-1 MAX cameras"""
    def __init__(
        self, 
        resolution: tuple[int, int], 
        fps: int,
        mx_id: str = None
    ):
        super().__init__(resolution, fps)
        self.mx_id = mx_id  # Device MX ID for specific camera selection

class StandardCameraConfig(CameraConfig):
    """Configuration for standard webcams"""
    def __init__(
        self, 
        resolution: tuple[int, int], 
        fps: int,
        device_id: int = 0
    ):
        super().__init__(resolution, fps)
        self.device_id = device_id

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Camera settings
    OAK_LEFT_RESOLUTION: tuple[int, int] = tuple(map(int, os.getenv("OAK_LEFT_RESOLUTION", "1280,720").split(",")))
    OAK_LEFT_FPS: int = int(os.getenv("OAK_LEFT_FPS", "30"))
    OAK_LEFT_MX_ID: str = os.getenv("OAK_LEFT_MX_ID", None)
    
    OAK_RIGHT_RESOLUTION: tuple[int, int] = tuple(map(int, os.getenv("OAK_RIGHT_RESOLUTION", "1280,720").split(",")))
    OAK_RIGHT_FPS: int = int(os.getenv("OAK_RIGHT_FPS", "30"))
    OAK_RIGHT_MX_ID: str = os.getenv("OAK_RIGHT_MX_ID", None)
    
    OBSBOT_RESOLUTION: tuple[int, int] = tuple(map(int, os.getenv("OBSBOT_RESOLUTION", "3840,2160").split(",")))
    OBSBOT_FPS: int = int(os.getenv("OBSBOT_FPS", "30"))
    OBSBOT_DEVICE_ID: int = int(os.getenv("OBSBOT_DEVICE_ID", "0"))
    
    # Stream settings
    STREAM_QUALITY: int = int(os.getenv("STREAM_QUALITY", "85"))  # JPEG quality 0-100
    
    def get_oak_left_config(self) -> OakCameraConfig:
        return OakCameraConfig(
            resolution=self.OAK_LEFT_RESOLUTION,
            fps=self.OAK_LEFT_FPS,
            mx_id=self.OAK_LEFT_MX_ID
        )
    
    def get_oak_right_config(self) -> OakCameraConfig:
        return OakCameraConfig(
            resolution=self.OAK_RIGHT_RESOLUTION,
            fps=self.OAK_RIGHT_FPS,
            mx_id=self.OAK_RIGHT_MX_ID
        )
    
    def get_obsbot_config(self) -> StandardCameraConfig:
        return StandardCameraConfig(
            resolution=self.OBSBOT_RESOLUTION,
            fps=self.OBSBOT_FPS,
            device_id=self.OBSBOT_DEVICE_ID
        )

# Create a singleton instance
settings = Settings()