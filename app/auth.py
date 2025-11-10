from fastapi import Depends, Header, HTTPException, status
from firebase_admin import auth
from firebase_admin import credentials, initialize_app
from google.cloud import secretmanager
from .config import settings

cred = credentials.Certificate(settings.firebase_cert_path)
#initialize_app(cred, name=settings.firebase_app_name)
initialize_app(cred)

def get_current_user(authorization: str = Header(...)):
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        decoded = auth.verify_id_token(token)
        return decoded
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
