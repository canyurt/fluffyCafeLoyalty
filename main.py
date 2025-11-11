#!/usr/bin/env python3
"""
Fluffy Café Loyalty - Complete Backend
Single file implementation with FastAPI + Firebase + Cloud Vision
"""

import os
import uuid
from datetime import datetime
from typing import Optional

import firebase_admin
from firebase_admin import auth as firebase_auth, credentials, firestore
from fastapi import FastAPI, Depends, Header, HTTPException, status, File, UploadFile, BackgroundTasks
from google.cloud import storage, vision
from pydantic import BaseModel
from pydantic_settings import BaseSettings


# ============================================================================
# CONFIGURATION
# ============================================================================

class Settings(BaseSettings):
    project_id: str
    firebase_cert_path: str = "/secrets/firebase-service-account.json"
    storage_receipts_bucket: str
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()


# ============================================================================
# FIREBASE INITIALIZATION
# ============================================================================

cred = credentials.Certificate(settings.firebase_cert_path)
firebase_admin.initialize_app(cred)
db = firestore.client()


# ============================================================================
# STORAGE & VISION CLIENTS
# ============================================================================

storage_client = storage.Client()
vision_client = vision.ImageAnnotatorClient()


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Fluffy Loyalty API", version="0.1.0")


# ============================================================================
# AUTH DEPENDENCY
# ============================================================================

def get_current_user(authorization: str = Header(...)):
    """Verify Firebase ID token from Authorization header"""
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Missing Bearer token"
        )
    token = authorization.split(" ", 1)[1]
    try:
        decoded = firebase_auth.verify_id_token(token)
        return decoded
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid token"
        ) from exc


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ProfileRequest(BaseModel):
    first_name: str
    last_name: str
    phone_number: Optional[str] = None
    date_of_birth: Optional[str] = None


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/healthz")
def healthcheck():
    """Health check endpoint"""
    return {"status": "ok"}


# ============================================================================
# PROFILE ENDPOINTS
# ============================================================================

@app.put("/v1/profile")
def upsert_profile(body: ProfileRequest, user=Depends(get_current_user)):
    """Create or update user profile"""
    uid = user["uid"]
    
    payload = {
        "user_id": uid,
        "email": user.get("email"),
        "first_name": body.first_name,
        "last_name": body.last_name,
        "phone_number": body.phone_number,
        "date_of_birth": body.date_of_birth,
        "updated_at": datetime.utcnow().isoformat(),
    }
    
    # Remove None values
    payload = {k: v for k, v in payload.items() if v is not None}
    
    # Upsert to Firestore
    doc_ref = db.collection("users").document(uid)
    doc_ref.set(payload, merge=True)
    
    return {"status": "ok"}


@app.get("/v1/profile")
def get_profile(user=Depends(get_current_user)):
    """Get current user profile"""
    uid = user["uid"]
    doc = db.collection("users").document(uid).get()
    return doc.to_dict() if doc.exists else {}


# ============================================================================
# RECEIPT UPLOAD ENDPOINT
# ============================================================================

@app.post("/v1/receipts")
async def upload_receipt_endpoint(
    background_tasks: BackgroundTasks,
    receipt: UploadFile = File(...),
    user=Depends(get_current_user),
):
    """
    Upload receipt image/PDF for processing
    Returns immediately with receipt_id while processing in background
    """
    
    # Validate content type
    if receipt.content_type not in {"image/jpeg", "image/png", "application/pdf"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only JPEG/PNG/PDF receipts are accepted."
        )
    
    uid = user["uid"]
    receipt_id = str(uuid.uuid4())
    
    # Read file contents
    contents = await receipt.read()
    
    # Upload to GCS
    bucket = storage_client.bucket(settings.storage_receipts_bucket)
    object_name = f"users/{uid}/receipts/{receipt_id}"
    blob = bucket.blob(object_name)
    blob.upload_from_string(contents, content_type=receipt.content_type)
    blob.metadata = {"user_id": uid}
    blob.patch()
    
    # Create initial Firestore record (SYNCHRONOUSLY - no await)
    initial_record = {
        "receipt_id": receipt_id,
        "user_id": uid,
        "gcs_object": object_name,
        "content_type": receipt.content_type,
        "status": "processing",
        "uploaded_at": datetime.utcnow().isoformat()
    }
    
    doc_ref = (
        db.collection("users")
        .document(uid)
        .collection("receipts")
        .document(receipt_id)
    )
    doc_ref.set(initial_record)
    
    # Schedule background processing
    def process_receipt_in_background():
        """Background task to extract OCR data"""
        try:
            # Run Vision API OCR
            gcs_uri = f"gs://{settings.storage_receipts_bucket}/{object_name}"
            image = vision.Image()
            image.source.image_uri = gcs_uri
            
            response = vision_client.text_detection(image=image)
            texts = response.text_annotations
            
            # Extract data
            full_text = texts[0].description if texts else ""
            
            # Simple parsing (you can enhance this)
            merchant_name = None
            total_amount = None
            
            lines = full_text.split("\n")
            for line in lines:
                line_upper = line.upper()
                if "FLUFFY" in line_upper or "CAFE" in line_upper:
                    merchant_name = line.strip()
                if "TOTAL" in line_upper or "$" in line or "€" in line:
                    # Try to extract number
                    import re
                    amounts = re.findall(r'\d+[.,]\d{2}', line)
                    if amounts:
                        total_amount = amounts[-1]  # Take last number
            
            # Update Firestore with results
            update_data = {
                "status": "ready",
                "merchant_name": merchant_name or "Unknown",
                "total_amount": total_amount or "0.00",
                "raw_text": full_text[:500],  # Truncate for storage
                "processed_at": datetime.utcnow().isoformat()
            }
            
            doc_ref.update(update_data)
            
        except Exception as e:
            # Log error and update status
            doc_ref.update({
                "status": "error",
                "error_message": str(e),
                "processed_at": datetime.utcnow().isoformat()
            })
    
    background_tasks.add_task(process_receipt_in_background)
    
    return {
        "receipt_id": receipt_id,
        "processing_status": "queued"
    }


# ============================================================================
# GET RECEIPTS
# ============================================================================

@app.get("/v1/receipts")
def get_receipts(user=Depends(get_current_user)):
    """Get all receipts for current user"""
    uid = user["uid"]
    
    receipts_ref = (
        db.collection("users")
        .document(uid)
        .collection("receipts")
        .order_by("uploaded_at", direction=firestore.Query.DESCENDING)
        .limit(50)
    )
    
    receipts = []
    for doc in receipts_ref.stream():
        receipts.append(doc.to_dict())
    
    return {"receipts": receipts}


@app.get("/v1/receipts/{receipt_id}")
def get_receipt(receipt_id: str, user=Depends(get_current_user)):
    """Get specific receipt details"""
    uid = user["uid"]
    
    doc = (
        db.collection("users")
        .document(uid)
        .collection("receipts")
        .document(receipt_id)
        .get()
    )
    
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Receipt not found")
    
    return doc.to_dict()


# ============================================================================
# MAIN (for local testing)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
