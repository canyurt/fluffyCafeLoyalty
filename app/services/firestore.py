from typing import Any, Mapping
from firebase_admin import firestore
from google.cloud.firestore_v1 import Client
from .config import settings

db: Client = firestore.client(app=None)

def upsert_user(uid: str, payload: Mapping[str, Any]):
    doc_ref = db.collection(settings.firestore_users_collection).document(uid)
    doc_ref.set(payload, merge=True)

def get_user(uid: str) -> dict | None:
    doc = db.collection(settings.firestore_users_collection).document(uid).get()
    return doc.to_dict() if doc.exists else None

def create_receipt(uid: str, receipt_id: str, payload: Mapping[str, Any]):
    doc_ref = (
        db.collection(settings.firestore_users_collection)
          .document(uid)
          .collection(settings.firestore_receipts_collection)
          .document(receipt_id)
    )
    doc_ref.set(payload)
