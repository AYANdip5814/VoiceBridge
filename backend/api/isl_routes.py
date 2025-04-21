from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from typing import Dict, List
import io
from ..services.isl_interpreter import ISLInterpreter

router = APIRouter()
interpreter = ISLInterpreter()

@router.post("/interpret")
async def interpret_frame(file: UploadFile = File(...)) -> Dict:
    """
    Interpret a single frame for ISL signs.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Dictionary containing interpretation results
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Interpret frame
        result = interpreter.interpret_frame(frame)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/state")
async def get_state() -> Dict:
    """Get the current interpretation state."""
    return interpreter.get_current_state()

@router.post("/reset")
async def reset_interpreter() -> Dict:
    """Reset the interpreter state."""
    interpreter.reset()
    return {"status": "success", "message": "Interpreter reset successfully"} 