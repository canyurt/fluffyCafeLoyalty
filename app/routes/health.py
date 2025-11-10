from fastapi import APIRouter

router = APIRouter()

@router.get("/healthz")
def healthcheck():
    return {"status": "ok"}
