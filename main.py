from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.v1 import endpoints
from api.services.inference_service import InferenceService
from api.core.config import settings

app = FastAPI(title="LeaderOracle API")

# CORS configuration: Allow traffic from localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://leader-oracle-ai.fly.dev/", "https://leader-oracle-ai.fly.dev"],  # Allow requests from localhost:3000
    allow_credentials=True,                   # Allow cookies to be sent
    allow_methods=["*"],                      # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],                      # Allow all headers
)

# Include API endpoints
app.include_router(endpoints.router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    # You can add any startup tasks here, such as loading the model
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)