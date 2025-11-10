from datetime import datetime
from typing import Any
from google.cloud import documentai
from .config import settings

def extract_fields_from_receipt(gcs_uri: str) -> dict[str, Any]:
    if not settings.documentai_processor_id:
        # stub fallback; replace with Vision or manual entry
        return {"status": "pending_manual_review"}
    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(settings.project_id,
                                 settings.documentai_location,
                                 settings.documentai_processor_id)
    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=gcs_uri.encode("utf-8"),
                                            mime_type="application/pdf")
    )
    # TODO: adapt when switching to GCS input (recommended).
    result = client.process_document(request=request)
    doc = result.document
    total = None
    merchant_name = None
    for ent in doc.entities:
        if ent.type_ == "total_amount":
            total = ent.mention_text
        if ent.type_ == "supplier_name":
            merchant_name = ent.mention_text
    return {
        "merchant_name": merchant_name,
        "total_amount": total,
        "processed_at": datetime.utcnow().isoformat(),
        "raw_entities": [e.to_json() for e in doc.entities],
    }
