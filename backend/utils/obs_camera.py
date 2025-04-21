import cv2
import numpy as np
from typing import Optional, Tuple
import subprocess
import os
import platform

class OBSCamera:
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.capture = None
        self.is_running = False
        
    def start(self) -> bool:
        """Start the OBS virtual camera capture."""
        try:
            # Try to find OBS virtual camera
            self.capture = cv2.VideoCapture(0)  # Usually OBS virtual cam is index 0
            
            # Set camera properties
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)
            
            if not self.capture.isOpened():
                raise Exception("Failed to open OBS virtual camera")
                
            self.is_running = True
            return True
            
        except Exception as e:
            print(f"Error starting OBS camera: {str(e)}")
            return False
            
    def stop(self):
        """Stop the OBS virtual camera capture."""
        if self.capture:
            self.capture.release()
        self.is_running = False
        
    def get_frame(self) -> Optional[np.ndarray]:
        """Get a frame from the OBS virtual camera."""
        if not self.is_running or not self.capture:
            return None
            
        ret, frame = self.capture.read()
        if not ret:
            return None
            
        return frame
        
    def is_available(self) -> bool:
        """Check if OBS virtual camera is available."""
        temp_capture = cv2.VideoCapture(0)
        is_available = temp_capture.isOpened()
        temp_capture.release()
        return is_available
        
    @staticmethod
    def check_obs_installation() -> bool:
        """Check if OBS is installed on the system."""
        system = platform.system().lower()
        
        if system == "windows":
            # Check common OBS installation paths on Windows
            obs_paths = [
                os.path.expandvars("%ProgramFiles%\\obs-studio"),
                os.path.expandvars("%ProgramFiles(x86)%\\obs-studio")
            ]
            return any(os.path.exists(path) for path in obs_paths)
            
        elif system == "linux":
            # Check if OBS is installed via package manager
            try:
                subprocess.run(["which", "obs"], check=True, capture_output=True)
                return True
            except subprocess.CalledProcessError:
                return False
                
        elif system == "darwin":  # macOS
            # Check common OBS installation path on macOS
            return os.path.exists("/Applications/OBS Studio.app")
            
        return False 