from fastapi import APIRouter, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
from app.services.camera_manager import CameraManager
from app.utils.exceptions import CameraError
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up templates
templates = Jinja2Templates(directory="app/templates")

# Create router
router = APIRouter(tags=["webcams"])

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page with webcam streams"""
    camera_manager = CameraManager.get_instance()
    camera_status = camera_manager.get_camera_status()
    
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "camera_status": camera_status
        }
    )

@router.get("/stream/{camera_id}")
async def stream_video(camera_id: str):
    """Stream video from specified camera"""
    camera_manager = CameraManager.get_instance()
    
    try:
        if camera_id == "oak_left":
            return StreamingResponse(
                camera_manager.get_oak_left_stream(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )
        elif camera_id == "obsbot":
            return StreamingResponse(
                camera_manager.get_obsbot_stream(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )
        elif camera_id == "oak_right":
            return StreamingResponse(
                camera_manager.get_oak_right_stream(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )
        else:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    except CameraError as e:
        logger.error(f"Camera error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

@router.get("/status")
async def get_status():
    """Get status of all cameras"""
    camera_manager = CameraManager.get_instance()
    return camera_manager.get_camera_status()