import uuid
from google.cloud import storage
from .config import settings

storage_client = storage.Client()
bucket = storage_client.bucket(settings.storage_receipts_bucket)

def upload_receipt(uid: str, file_bytes: bytes, content_type: str) -> str:
    object_name = f"users/{uid}/receipts/{uuid.uuid4()}"
    blob = bucket.blob(object_name)
    blob.upload_from_string(file_bytes, content_type=content_type)
    blob.metadata = {"user_id": uid}
    blob.patch()
    return object_name

def get_signed_url(object_name: str, minutes: int = 60) -> str:
    blob = bucket.blob(object_name)
    return blob.generate_signed_url(expiration=minutes * 60)
