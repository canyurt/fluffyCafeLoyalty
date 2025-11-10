from datetime import datetime
from fastapi import APIRouter, Depends
from pydantic import BaseModel, EmailStr
from ..auth import get_current_user
from ..services.firestore import upsert_user, get_user

router = APIRouter()

class ProfileRequest(BaseModel):
    first_name: str
    last_name: str
    phone_number: str | None = None
    date_of_birth: str | None = None

@router.put("/profile")
def upsert_profile(body: ProfileRequest, user=Depends(get_current_user)):
    uid = user["uid"]
    payload = {
        "user_id": uid,
        "email": user.get("email"),
        "first_name": body.first_name,
        "last_name": body.last_name,
        "phone_number": body.phone_number,
        "date_of_birth": body.date_of_birth,
        "updated_at": datetime.utcnow().isoformat(),
        "created_at": user.get("firebase", {}).get("sign_in_provider") == "custom"
            and datetime.utcnow().isoformat()
            or None,
    }
    upsert_user(uid, {k: v for k, v in payload.items() if v is not None})
    return {"status": "ok"}

@router.get("/profile")
def get_profile(user=Depends(get_current_user)):
    uid = user["uid"]
    doc = get_user(uid)
    return doc or {}
