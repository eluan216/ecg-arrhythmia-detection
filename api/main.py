"""Main API application entry point."""

import os

from fastapi.middleware.cors import CORSMiddleware

from api import app as base_app

app = base_app

# Explicit origins only — override with ECG_CORS_ORIGINS=comma,separated,list
_raw = os.getenv("ECG_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
_origins = [o.strip() for o in _raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
