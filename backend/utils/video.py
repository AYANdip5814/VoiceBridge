import cv2
from typing import List, Dict

def get_video_sources() -> List[Dict[str, str]]:
    """Get a list of available video sources."""
    sources = []
    
    # Try to get the number of cameras
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        
        # Get camera name (platform dependent)
        name = f"Camera {index}"
        sources.append({
            'id': str(index),
            'name': name
        })
        
        cap.release()
        index += 1
    
    return sources

def get_frame_dimensions(source_id: str) -> Dict[str, int]:
    """Get the dimensions of frames from a video source."""
    cap = cv2.VideoCapture(int(source_id))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return {
        'width': width,
        'height': height
    }

def resize_frame(frame, target_width: int, target_height: int):
    """Resize a frame while maintaining aspect ratio."""
    height, width = frame.shape[:2]
    aspect = width / height
    
    if width > height:
        new_width = target_width
        new_height = int(target_width / aspect)
    else:
        new_height = target_height
        new_width = int(target_height * aspect)
        
    return cv2.resize(frame, (new_width, new_height)) 