import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes import webcam_routes
from app.config import settings
from app.services.camera_manager import CameraManager

app = FastAPI(
    title="Webcam Viewer",
    description="A FastAPI application to view and control multiple webcams",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routers
app.include_router(webcam_routes.router)

# Initialize cameras on startup and close on shutdown
@app.on_event("startup")
async def startup_event():
    camera_manager = CameraManager.get_instance()
    await camera_manager.initialize_cameras()

@app.on_event("shutdown")
async def shutdown_event():
    camera_manager = CameraManager.get_instance()
    await camera_manager.close_all_cameras()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", 
        host=settings.HOST, 
        port=settings.PORT,
        reload=settings.DEBUG
    )