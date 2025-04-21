from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import isl_routes

app = FastAPI(
    title="VoiceBridge API",
    description="Real-time ISL interpretation API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(isl_routes.router, prefix="/api/isl", tags=["ISL Interpretation"])

@app.get("/")
async def root():
    return {"message": "Welcome to VoiceBridge API"} 