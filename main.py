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
            # # Run Vision API OCR
            # gcs_uri = f"gs://{settings.storage_receipts_bucket}/{object_name}"
            # image = vision.Image()
            # image.source.image_uri = gcs_uri
            
            # response = vision_client.text_detection(image=image)
            # texts = response.text_annotations
            
            # # Extract data
            # full_text = texts[0].description if texts else ""
            
            # # Simple parsing (you can enhance this)
            # merchant_name = None
            # total_amount = None
            
            # lines = full_text.split("\n")
            # for line in lines:
            #     line_upper = line.upper()
            #     if "FLUFFY" in line_upper or "CAFE" in line_upper:
            #         merchant_name = line.strip()
            #     if "TOTAL" in line_upper or "$" in line or "€" in line:
            #         # Try to extract number
            #         import re
            #         amounts = re.findall(r'\d+[.,]\d{2}', line)
            #         if amounts:
            #             total_amount = amounts[-1]  # Take last number
            
            # # Update Firestore with results
            # update_data = {
            #     "status": "ready",
            #     "merchant_name": merchant_name or "Unknown",
            #     "total_amount": total_amount or "0.00",
            #     "raw_text": full_text[:500],  # Truncate for storage
            #     "processed_at": datetime.utcnow().isoformat()
            # }
            
            # doc_ref.update(update_data)
            
            # Run Vision API OCR
            gcs_uri = f"gs://{settings.storage_receipts_bucket}/{object_name}"
            image = vision.Image()
            image.source.image_uri = gcs_uri
            response = vision_client.text_detection(image=image)
            texts = response.text_annotations
            
            full_text = texts[0].description if texts else ""
            
            # NEW: Convert to structured schema + markdown
            receipt_data, markdown_text = parse_receipt_text(full_text)
            
            update_data = {
                "status": "ready",
                "processed_at": datetime.utcnow().isoformat(),
                "raw_text": full_text[:1500],  # store raw for debugging
                "receipt_data": receipt_data,   # structured JSON
                "markdown_view": markdown_text, # markdown formatted receipt
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




# import re

# def parse_receipt_text(full_text: str):
#     """
#     Convert OCR text into structured data (schema) + markdown view.
#     Assumes receipt layout is fixed (Fluffy Café receipts).
#     """

#     lines = [line.strip() for line in full_text.split("\n") if line.strip()]

#     # Extract header & metadata
#     store_name = lines[1]
#     terminal_info = lines[2]
#     statement = lines[3]
#     statement_no = statement.replace("Statement", "").strip()

#     # Extract date/time (regex matches dd/mm/yyyy, hh:mm)
#     datetime_match = re.search(r"(\d{2}/\d{2}/\d{4}),\s*(\d{2}:\d{2})", full_text)
#     date = datetime_match.group(1) if datetime_match else ""
#     time = datetime_match.group(2) if datetime_match else ""

#     # Extract table
#     table_line = next((l for l in lines if "Tafel" in l or "Table" in l), None)
#     table = table_line.replace("Floor plan", "").strip() if table_line else ""

#     # Extract tax ID
#     tax_id_match = re.search(r"Btw[: ]+([A-Z0-9]+)", full_text)
#     tax_id = tax_id_match.group(1) if tax_id_match else None

#     # Extract items by identifying price-aligned lines
#     items = []
#     for line in lines:
#         m = re.match(r"(.+?)\s+(\d+[.,]\d{2})$", line)
#         if m:
#             name = m.group(1).strip()
#             price = float(m.group(2).replace(",", "."))
#             items.append({"name": name, "price": price})

#     total = sum(item["price"] for item in items)

#     # Build structured schema
#     data = {
#         "store_name": store_name,
#         "statement_id": statement_no,
#         "date": date,
#         "time": time,
#         "table": table,
#         "items": items,
#         "subtotal": total,
#         "tax_id": tax_id,
#     }

#     # Build markdown representation
#     markdown = f"# {store_name}\n\n"
#     markdown += f"**Date:** {date} {time}\n"
#     markdown += f"**Table:** {table}\n\n---\n\n"
#     markdown += "| Item | Price (€) |\n|------|-----------:|\n"
#     for item in items:
#         markdown += f"| {item['name']} | {item['price']:.2f} |\n"
#     markdown += f"| **Total** | **{total:.2f}** |\n\n"
#     markdown += f"**Tax ID:** {tax_id}\n"

#     return data, markdown
# import re
# from typing import List, Dict, Tuple  # add at top of file if not already present


# def parse_receipt_text(full_text: str) -> Tuple[Dict, str]:
#     """
#     Convert OCR text from Fluffy Café receipts into structured data + markdown.
#     """
#     text = (full_text or "").replace("\r", "")
#     text_compact = re.sub(r"\s+", " ", text).strip()

#     # Store name
#     store_match = re.search(r"(Fluffy\s+Cafe-?Restaur[a-z]*)", text, re.IGNORECASE)
#     store_name = store_match.group(1).strip() if store_match else None

#     # Order device / employee number (e.g. "iPad1/696233-Merve")
#     terminal_match = re.search(r"([A-Za-z][A-Za-z0-9]*)\s*/\s*([A-Za-z0-9\-]+)", text)
#     order_device = terminal_match.group(1) if terminal_match else None
#     employee_no = terminal_match.group(2) if terminal_match else None

#     # Statement number (e.g. "Statement N939388.21511")
#     statement_match = re.search(r"Statement\s+([A-Z0-9.\-]+)", text, re.IGNORECASE)
#     statement_number = statement_match.group(1) if statement_match else None

#     # Date / time
#     datetime_match = re.search(r"(\d{2}/\d{2}/\d{4}),\s*(\d{2}:\d{2})", text)
#     date = datetime_match.group(1) if datetime_match else ""
#     time = datetime_match.group(2) if datetime_match else ""

#     # Table number (only the numeric part)
#     table_match = re.search(r"(?:Tafel|Table)\s+(\d+)", text, re.IGNORECASE)
#     table = table_match.group(1) if table_match else None
#     table_end_idx = table_match.end() if table_match else 0

#     # Segment containing item names (between table info and "Amount due")
#     amount_due_idx = text.lower().find("amount due")
#     if amount_due_idx == -1:
#         amount_due_idx = text.lower().find("amount")
#     items_block = text[table_end_idx:amount_due_idx] if amount_due_idx > table_end_idx else text[table_end_idx:]
#     items_block = re.sub(r"Floor plan[^,]*,\s*(?:Tafel|Table)\s+\d+", "", items_block, flags=re.IGNORECASE)
#     items_block = re.sub(r"\s+", " ", items_block).strip()
#     items_block = re.sub(r"(?<=[a-z])(?=[A-Z])", "\n", items_block)
#     item_names: List[str] = [name.strip(" ,") for name in items_block.split("\n") if name.strip()]

#     # Segment containing monetary values (between "Amount" and the euro total)
#     euro_idx = text.find("€", amount_due_idx if amount_due_idx != -1 else 0)
#     if euro_idx == -1:
#         euro_idx = len(text)
#     price_region = text[amount_due_idx if amount_due_idx != -1 else table_end_idx:euro_idx]
#     price_values = re.findall(r"\d+[.,]\d{2}", price_region)
#     if item_names and len(price_values) >= len(item_names):
#         price_values = price_values[-len(item_names):]
#     item_prices: List[float] = [float(value.replace(",", ".")) for value in price_values]

#     items: List[Dict] = []
#     for idx, name in enumerate(item_names):
#         price = item_prices[idx] if idx < len(item_prices) else None
#         items.append({"name": name, "price": price})

#     # Total amount from the € line
#     total_match = re.search(r"€\s*([\d.,]+)", text)
#     total_amount = float(total_match.group(1).replace(",", ".")) if total_match else (
#         round(sum(p for p in item_prices if p is not None), 2) if item_prices else None
#     )

#     # Tax ID
#     tax_match = re.search(r"Btw[: ]+([A-Z0-9]+)", text, re.IGNORECASE)
#     tax_id = tax_match.group(1) if tax_match else None

#     receipt_data = {
#         "store_name": store_name,
#         "statement_number": statement_number,
#         "order_device": order_device,
#         "employee_no": employee_no,
#         "date": date,
#         "time": time,
#         "table": table,
#         "items": items,
#         "total_amount": total_amount,
#         "tax_id": tax_id,
#     }

#     # Markdown view
#     markdown_lines = [f"# {store_name or 'Fluffy Cafe-Restaurant'}", ""]
#     if date or time:
#         markdown_lines.append(f"**Date:** {date} {time}".strip())
#     if table:
#         markdown_lines.append(f"**Table:** {table}")
#     if statement_number:
#         markdown_lines.append(f"**Statement:** {statement_number}")
#     if order_device or employee_no:
#         metadata_parts = []
#         if order_device:
#             metadata_parts.append(f"Device: {order_device}")
#         if employee_no:
#             metadata_parts.append(f"Employee: {employee_no}")
#         markdown_lines.append("**Assignment:** " + " | ".join(metadata_parts))
#     markdown_lines.append("\n---\n")
#     markdown_lines.append("| Item | Price (€) |\n|------|-----------:|")
#     for item in items:
#         price_text = f"{item['price']:.2f}" if item.get("price") is not None else ""
#         markdown_lines.append(f"| {item['name']} | {price_text} |")
#     if total_amount is not None:
#         markdown_lines.append(f"| **Total** | **{total_amount:.2f}** |")
#     markdown_lines.append("")
#     if tax_id:
#         markdown_lines.append(f"**Tax ID:** {tax_id}")

#     markdown_text = "\n".join(markdown_lines)

#     return receipt_data, markdown_text

# import re
# from typing import List, Dict, Tuple  # ensure this import is present once near the top of the file


# def parse_receipt_text(full_text: str) -> Tuple[Dict, str]:
#     """
#     Convert OCR text from Fluffy Café receipts into structured data + markdown.
#     """
#     text = (full_text or "").replace("\r", "")
#     text_lower = text.lower()
#     text_compact = re.sub(r"\s+", " ", text).strip()

#     # Store name
#     store_match = re.search(r"(Fluffy\s+Cafe-?Restaur[a-z]*)", text, re.IGNORECASE)
#     store_name = store_match.group(1).strip() if store_match else None

#     # Order device / employee number (e.g. "iPad1/696233-Merve")
#     terminal_match = re.search(r"([A-Za-z][A-Za-z0-9]*)\s*/\s*([A-Za-z0-9\-]+)", text)
#     order_device = terminal_match.group(1) if terminal_match else None
#     employee_no = terminal_match.group(2) if terminal_match else None

#     # Statement number (e.g. "Statement N939388.21511")
#     statement_match = re.search(r"Statement\s+([A-Z0-9.\-]+)", text, re.IGNORECASE)
#     statement_number = statement_match.group(1) if statement_match else None

#     # Date / time
#     datetime_match = re.search(r"(\d{2}/\d{2}/\d{4}),\s*(\d{2}:\d{2})", text)
#     date = datetime_match.group(1) if datetime_match else ""
#     time = datetime_match.group(2) if datetime_match else ""

#     # Table number (only the numeric part)
#     table_match = re.search(r"(?:Tafel|Table)\s+(\d+)", text, re.IGNORECASE)
#     table = table_match.group(1) if table_match else None
#     table_end_idx = table_match.end() if table_match else 0

#     # Segment containing menu items (between table info and "Amount …")
#     amount_due_idx = text_lower.find("amount due")
#     if amount_due_idx == -1:
#         amount_due_idx = text_lower.find("amount")
#     if amount_due_idx == -1:
#         amount_due_idx = len(text)

#     items_section = text[table_end_idx:amount_due_idx] if table_end_idx < amount_due_idx else ""
#     items_section = re.sub(r"Floor plan[^,]*,\s*(?:Tafel|Table)\s+\d+", "", items_section, flags=re.IGNORECASE)
#     items_section = re.sub(r"\s+", " ", items_section).strip()

#     item_pattern = re.compile(r"[A-Z][\w'&-]*(?:\s+[A-Z][\w'&-]*)*")
#     raw_names = item_pattern.findall(items_section)
#     blacklist = {"", "Floor", "Plan"}
#     item_names: List[str] = []
#     for candidate in raw_names:
#         if candidate in blacklist or candidate.lower().startswith("floor plan"):
#             continue
#         if candidate not in item_names:
#             item_names.append(candidate.strip())

#     # Monetary values (prefer region after “Btw” before the € total)
#     euro_idx = text.find("€")
#     price_region_start = text_lower.find("btw")
#     if price_region_start == -1:
#         price_region_start = amount_due_idx
#     price_region = text[price_region_start:euro_idx] if euro_idx > price_region_start >= 0 else text[price_region_start:]
#     price_values = re.findall(r"\d+[.,]\d{2}", price_region)

#     if item_names and len(price_values) < len(item_names):
#         # Fallback: grab trailing monetary values from entire text
#         price_values = re.findall(r"\d+[.,]\d{2}", text)
#     if item_names and len(price_values) >= len(item_names):
#         price_values = price_values[-len(item_names):]

#     item_prices = [float(v.replace(",", ".")) for v in price_values[:len(item_names)]]
#     while len(item_prices) < len(item_names):
#         item_prices.append(None)

#     items: List[Dict] = []
#     for idx, name in enumerate(item_names):
#         price = item_prices[idx] if idx < len(item_prices) else None
#         items.append({"name": name, "price": price})

#     # Total amount from the € line
#     total_match = re.search(r"€\s*([\d.,]+)", text)
#     total_amount = float(total_match.group(1).replace(",", ".")) if total_match else (
#         round(sum(p for p in item_prices if p is not None), 2) if item_prices else None
#     )

#     # Tax ID
#     tax_match = re.search(r"Btw[: ]+([A-Z0-9]+)", text, re.IGNORECASE)
#     tax_id = tax_match.group(1) if tax_match else None

#     receipt_data = {
#         "store_name": store_name,
#         "statement_number": statement_number,
#         "order_device": order_device,
#         "employee_no": employee_no,
#         "date": date,
#         "time": time,
#         "table": table,
#         "items": items,
#         "total_amount": total_amount,
#         "tax_id": tax_id,
#     }

#     # Markdown view
#     markdown_lines = [f"# {store_name or 'Fluffy Cafe-Restaurant'}", ""]
#     if date or time:
#         markdown_lines.append(f"**Date:** {date} {time}".strip())
#     if table:
#         markdown_lines.append(f"**Table:** {table}")
#     if statement_number:
#         markdown_lines.append(f"**Statement:** {statement_number}")
#     if order_device or employee_no:
#         metadata_parts = []
#         if order_device:
#             metadata_parts.append(f"Device: {order_device}")
#         if employee_no:
#             metadata_parts.append(f"Employee: {employee_no}")
#         markdown_lines.append("**Assignment:** " + " | ".join(metadata_parts))
#     markdown_lines.append("\n---\n")
#     markdown_lines.append("| Item | Price (€) |\n|------|-----------:|")
#     for item in items:
#         price_text = f"{item['price']:.2f}" if item.get("price") is not None else ""
#         markdown_lines.append(f"| {item['name']} | {price_text} |")
#     if total_amount is not None:
#         markdown_lines.append(f"| **Total** | **{total_amount:.2f}** |")
#     markdown_lines.append("")
#     if tax_id:
#         markdown_lines.append(f"**Tax ID:** {tax_id}")

#     markdown_text = "\n".join(markdown_lines)

#     return receipt_data, markdown_text

# import re
# from typing import List, Dict, Tuple  # ensure this import appears only once near the top of the file


# def parse_receipt_text(full_text: str) -> Tuple[Dict, str]:
#     """
#     Convert OCR text from Fluffy Café receipts into structured data + markdown.
#     """
#     text = (full_text or "").replace("\r", "")
#     lines = [line.strip() for line in text.splitlines() if line.strip()]
#     lines_lower = [line.lower() for line in lines]

#     # Store name
#     store_name = next((line for line in lines if re.search(r"fluffy\s+cafe", line, re.IGNORECASE)), None)

#     # Order device / employee number (e.g. "iPad1/696233-Merve")
#     terminal_line = next((line for line in lines if "/" in line and "statement" not in line.lower()), None)
#     order_device = None
#     employee_no = None
#     if terminal_line:
#         terminal_match = re.search(r"([A-Za-z][A-Za-z0-9]*)\s*/\s*([A-Za-z0-9\-]+)", terminal_line)
#         if terminal_match:
#             order_device = terminal_match.group(1)
#             employee_no = terminal_match.group(2)

#     # Statement number (e.g. "Statement N939388.21511")
#     statement_line = next((line for line in lines if line.lower().startswith("statement")), None)
#     statement_number = None
#     if statement_line:
#         statement_match = re.search(r"Statement\s+([A-Z0-9.\-]+)", statement_line, re.IGNORECASE)
#         if statement_match:
#             statement_number = statement_match.group(1)

#     # Date / time
#     datetime_line = next((line for line in lines if re.search(r"\d{2}/\d{2}/\d{4}", line)), "")
#     datetime_match = re.search(r"(\d{2}/\d{2}/\d{4}),\s*(\d{2}:\d{2})", datetime_line)
#     date = datetime_match.group(1) if datetime_match else ""
#     time = datetime_match.group(2) if datetime_match else ""

#     # Table number (only the numeric part)
#     table_idx = next((idx for idx, line in enumerate(lines) if re.search(r"(?:Tafel|Table)\s+\d+", line, re.IGNORECASE)), -1)
#     table = None
#     if table_idx != -1:
#         table_match = re.search(r"(?:Tafel|Table)\s+(\d+)", lines[table_idx], re.IGNORECASE)
#         if table_match:
#             table = table_match.group(1)

#     # Find "Amount" line to delimit the items block
#     amount_idx = next((idx for idx, line in enumerate(lines_lower) if line.startswith("amount")), len(lines))

#     # Extract item lines between table info and "Amount …"
#     items_block_lines = lines[table_idx + 1:amount_idx] if table_idx != -1 else lines[:amount_idx]

#     items: List[Dict] = []
#     pending_name: str | None = None

#     for raw_line in items_block_lines:
#         line = raw_line.strip(":- ")
#         if not line or "plan" in line.lower():
#             continue

#         match_inline = re.match(r"(.+?)\s+(\d+[.,]\d{2})$", line)
#         if match_inline:
#             name = match_inline.group(1).strip()
#             price = float(match_inline.group(2).replace(",", "."))
#             items.append({"name": name, "price": price})
#             pending_name = None
#             continue

#         match_price_only = re.fullmatch(r"(\d+[.,]\d{2})", line)
#         if match_price_only and pending_name:
#             price = float(match_price_only.group(1).replace(",", "."))
#             items.append({"name": pending_name, "price": price})
#             pending_name = None
#             continue

#         # Otherwise treat as a name line; price may follow on next line
#         pending_name = line

#     # If a name was pending without a matched price, add it with price None
#     if pending_name:
#         items.append({"name": pending_name, "price": None})

#     # Total amount from the “€ …” line
#     total_line = next((line for line in lines if "€" in line), "")
#     total_match = re.search(r"€\s*([\d.,]+)", total_line)
#     total_amount = float(total_match.group(1).replace(",", ".")) if total_match else (
#         round(sum(item["price"] for item in items if item["price"] is not None), 2) if items else None
#     )

#     # Tax ID
#     tax_line = next((line for line in lines if line.lower().startswith("btw")), "")
#     tax_match = re.search(r"Btw[: ]+([A-Z0-9]+)", tax_line, re.IGNORECASE)
#     tax_id = tax_match.group(1) if tax_match else None

#     receipt_data = {
#         "store_name": store_name,
#         "statement_number": statement_number,
#         "order_device": order_device,
#         "employee_no": employee_no,
#         "date": date,
#         "time": time,
#         "table": table,
#         "items": items,
#         "total_amount": total_amount,
#         "tax_id": tax_id,
#     }

#     # Markdown representation
#     markdown_lines = [f"# {store_name or 'Fluffy Cafe-Restaurant'}", ""]
#     if date or time:
#         markdown_lines.append(f"**Date:** {date} {time}".strip())
#     if table:
#         markdown_lines.append(f"**Table:** {table}")
#     if statement_number:
#         markdown_lines.append(f"**Statement:** {statement_number}")
#     assignments = []
#     if order_device:
#         assignments.append(f"Device: {order_device}")
#     if employee_no:
#         assignments.append(f"Employee: {employee_no}")
#     if assignments:
#         markdown_lines.append("**Assignment:** " + " | ".join(assignments))
#     markdown_lines.append("\n---\n")
#     markdown_lines.append("| Item | Price (€) |\n|------|-----------:|")
#     for item in items:
#         price_text = f"{item['price']:.2f}" if item.get("price") is not None else ""
#         markdown_lines.append(f"| {item['name']} | {price_text} |")
#     if total_amount is not None:
#         markdown_lines.append(f"| **Total** | **{total_amount:.2f}** |")
#     markdown_lines.append("")
#     if tax_id:
#         markdown_lines.append(f"**Tax ID:** {tax_id}")

#     markdown_text = "\n".join(markdown_lines)

#     return receipt_data, markdown_text


import re
from typing import List, Dict, Optional, Tuple


def parse_receipt_text(full_text: str) -> Tuple[Dict, str]:
    """
    Convert OCR text into structured data (schema) + markdown view.
    Assumes receipt layout is fixed (Fluffy Café receipts).
    """
    text = (full_text or "").replace("\r", "")
    store_match = re.search(r"(Fluffy\s+Cafe-?Restaur[a-z]*)", text, re.IGNORECASE)
    store_name = (
        re.sub(r"\s+", " ", store_match.group(1)).strip()
        if store_match
        else "Fluffy Cafe-Restaurant"
    )

    terminal_match = re.search(r"([A-Za-z][A-Za-z0-9]*)\s*/\s*([A-Za-z0-9\-]+)", text)
    order_device = terminal_match.group(1).strip() if terminal_match else None
    employee_no = terminal_match.group(2).strip() if terminal_match else None

    statement_match = re.search(r"Statement\s+([A-Z0-9.\-]+)", text, re.IGNORECASE)
    statement_number = statement_match.group(1) if statement_match else None

    datetime_match = re.search(r"(\d{2}/\d{2}/\d{4}),\s*(\d{2}:\d{2})", text)
    date = datetime_match.group(1) if datetime_match else ""
    time = datetime_match.group(2) if datetime_match else ""

    table_match = re.search(r"(?:Tafel|Table)\s*(\d+)", text, re.IGNORECASE)
    table = table_match.group(1) if table_match else None

    # isolate the segment containing the item names
    items_region_start = table_match.end() if table_match else 0
    items_region = text[items_region_start:]
    amount_match = re.search(r"Amount\s+due", items_region, re.IGNORECASE)
    if not amount_match:
        amount_match = re.search(r"Amount", items_region, re.IGNORECASE)
    if amount_match:
        items_region = items_region[:amount_match.start()]

    items_region = (
        items_region.replace("\n", " ")
        .replace("\t", " ")
        .replace("\r", " ")
        .replace("•", " ")
    )
    items_region = re.sub(r"\s+", " ", items_region).strip(" ,-:")

    def split_items(segment: str) -> List[str]:
        tokens: List[str] = []
        buffer: List[str] = []
        for idx, ch in enumerate(segment):
            buffer.append(ch)
            next_char = segment[idx + 1] if idx + 1 < len(segment) else ""
            if ch.islower() and next_char.isupper():
                token = "".join(buffer).strip(" ,-:")
                if token:
                    tokens.append(token)
                buffer = []
        residual = "".join(buffer).strip(" ,-:")
        if residual:
            tokens.append(residual)
        return tokens

    raw_tokens = split_items(items_region)
    item_names: List[str] = []
    for token in raw_tokens:
        cleaned = token.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if any(
            keyword in lowered
            for keyword in ("floor plan", "amount", "division", "payment", "btw", "tax")
        ):
            continue
        if lowered.startswith(("table", "taf")):
            continue
        item_names.append(cleaned)

    price_pattern = re.compile(r"\d+[.,]\d{2}")
    total_match = re.search(r"€\s*([\d.,]+)", text)
    total_amount = float(total_match.group(1).replace(",", ".")) if total_match else None
    total_idx = total_match.start() if total_match else len(text)

    price_matches_zone = [
        m
        for m in price_pattern.finditer(text)
        if m.start() >= items_region_start and m.start() < total_idx
    ]
    price_values_zone = [float(m.group().replace(",", ".")) for m in price_matches_zone]
    if item_names and len(price_values_zone) >= len(item_names):
        price_values = price_values_zone[-len(item_names):]
    else:
        price_values = price_values_zone

    items: List[Dict] = []
    for idx, name in enumerate(item_names):
        price_value: Optional[float] = None
        if idx < len(price_values):
            price_value = round(price_values[idx], 2)
            # Round to two decimals consistently
            price_value = float(f"{price_value:.2f}")
        items.append({"name": name, "price": price_value})

    if total_amount is None and items:
        summed = sum(item["price"] for item in items if item["price"] is not None)
        total_amount = round(summed, 2) if summed else None

    tax_match = re.search(r"Btw[: ]+([A-Z0-9]+)", text, re.IGNORECASE)
    tax_id = tax_match.group(1) if tax_match else None

    receipt_data = {
        "store_name": store_name,
        "statement_number": statement_number,
        "order_device": order_device,
        "employee_no": employee_no,
        "date": date,
        "time": time,
        "table": table,
        "items": items,
        "total_amount": total_amount,
        "tax_id": tax_id,
    }

    markdown_lines = [f"# {store_name}", ""]
    if date or time:
        markdown_lines.append(f"**Date:** {date} {time}".strip())
    if table:
        markdown_lines.append(f"**Table:** {table}")
    if statement_number:
        markdown_lines.append(f"**Statement:** {statement_number}")
    assignment_parts = []
    if order_device:
        assignment_parts.append(f"Device: {order_device}")
    if employee_no:
        assignment_parts.append(f"Employee: {employee_no}")
    if assignment_parts:
        markdown_lines.append("**Assignment:** " + " | ".join(assignment_parts))
    markdown_lines.append("\n---\n")
    markdown_lines.append("| Item | Price (€) |\n|------|-----------:|")
    for item in items:
        price_text = f"{item['price']:.2f}" if item["price"] is not None else ""
        markdown_lines.append(f"| {item['name']} | {price_text} |")
    if total_amount is not None:
        markdown_lines.append(f"| **Total** | **{total_amount:.2f}** |")
    markdown_lines.append("")
    if tax_id:
        markdown_lines.append(f"**Tax ID:** {tax_id}")
    markdown_text = "\n".join(markdown_lines)

    return receipt_data, markdown_text






































# ============================================================================
# ============================================================================
# ============================================================================
#
# API SECTION
#
# ============================================================================
# ============================================================================
# ============================================================================






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
