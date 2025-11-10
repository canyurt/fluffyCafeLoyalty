import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, BackgroundTasks
from ..auth import get_current_user
from ..services.firestore import create_receipt
from ..services.storage import upload_receipt
from ..services.receipt_pipeline import extract_fields_from_receipt

router = APIRouter()

@router.post("/receipts")
async def upload_receipt(
    background_tasks: BackgroundTasks,
    receipt: UploadFile = File(...),
    user=Depends(get_current_user),
):
    if receipt.content_type not in {"image/jpeg", "image/png", "application/pdf"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only JPEG/PNG/PDF receipts are accepted."
        )
    contents = await receipt.read()
    object_path = upload_receipt(user["uid"], contents, receipt.content_type)
    receipt_id = str(uuid.uuid4())
    record = {
        "receipt_id": receipt_id,
        "user_id": user["uid"],
        "gcs_object": object_path,
        "content_type": receipt.content_type,
        "status": "processing",
        "uploaded_at": datetime.utcnow().isoformat()
    }
    create_receipt(user["uid"], receipt_id, record)

    def background_processing():
        parsed = extract_fields_from_receipt(f"gs://{object_path}")
        create_receipt(
            user["uid"],
            receipt_id,
            {**record, **parsed, "status": "ready"}
        )

    background_tasks.add_task(background_processing)
    return {"receipt_id": receipt_id, "processing_status": "queued"}
