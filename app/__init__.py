from fastapi import FastAPI

from .routes.health import router as health_router
from .routes.profile import router as profile_router
from .routes.receipts import router as receipts_router

def create_app() -> FastAPI:
    app = FastAPI(title="Fluffy Loyalty API", version="0.1.0")
    app.include_router(health_router, tags=["health"])
    app.include_router(profile_router, prefix="/v1", tags=["profile"])
    app.include_router(receipts_router, prefix="/v1", tags=["receipts"])
    return app

app = create_app()
