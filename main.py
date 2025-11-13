#!/usr/bin/env python3
"""
Fluffy Café Loyalty - Complete Backend
Single file implementation with FastAPI + Firebase + Cloud Vision
"""

import os
import uuid
from datetime import datetime
from typing import Optional
import statistics

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
            print(f"gcs_uri:{gcs_uri}")
            image = vision.Image()
            image.source.image_uri = gcs_uri
            # response = vision_client.text_detection(image=image)
            # With this line (Document Text Detection keeps structure)
            response = vision_client.document_text_detection(image=image)            
            texts = response.text_annotations
            print(f"texts:{texts}")
            
            full_text = texts[0].description if texts else ""
            print(f"full_text:{full_text}")            
            
            # NEW: Convert to structured schema + markdown
            # receipt_data, markdown_text = parse_receipt_text(full_text)
            receipt_data, markdown_text = parse_receipt_text(full_text, response=response)
            
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


# import re
# from typing import List, Dict, Optional, Tuple


# def parse_receipt_text(full_text: str) -> Tuple[Dict, str]:
#     """
#     Convert OCR text into structured data (schema) + markdown view.
#     Assumes receipt layout is fixed (Fluffy Café receipts).
#     """
#     text = (full_text or "").replace("\r", "")
#     store_match = re.search(r"(Fluffy\s+Cafe-?Restaur[a-z]*)", text, re.IGNORECASE)
#     store_name = (
#         re.sub(r"\s+", " ", store_match.group(1)).strip()
#         if store_match
#         else "Fluffy Cafe-Restaurant"
#     )

#     terminal_match = re.search(r"([A-Za-z][A-Za-z0-9]*)\s*/\s*([A-Za-z0-9\-]+)", text)
#     order_device = terminal_match.group(1).strip() if terminal_match else None
#     employee_no = terminal_match.group(2).strip() if terminal_match else None

#     statement_match = re.search(r"Statement\s+([A-Z0-9.\-]+)", text, re.IGNORECASE)
#     statement_number = statement_match.group(1) if statement_match else None

#     datetime_match = re.search(r"(\d{2}/\d{2}/\d{4}),\s*(\d{2}:\d{2})", text)
#     date = datetime_match.group(1) if datetime_match else ""
#     time = datetime_match.group(2) if datetime_match else ""

#     table_match = re.search(r"(?:Tafel|Table)\s*(\d+)", text, re.IGNORECASE)
#     table = table_match.group(1) if table_match else None

#     # isolate the segment containing the item names
#     items_region_start = table_match.end() if table_match else 0
#     items_region = text[items_region_start:]
#     amount_match = re.search(r"Amount\s+due", items_region, re.IGNORECASE)
#     if not amount_match:
#         amount_match = re.search(r"Amount", items_region, re.IGNORECASE)
#     if amount_match:
#         items_region = items_region[:amount_match.start()]

#     items_region = (
#         items_region.replace("\n", " ")
#         .replace("\t", " ")
#         .replace("\r", " ")
#         .replace("•", " ")
#     )
#     items_region = re.sub(r"\s+", " ", items_region).strip(" ,-:")

#     def split_items(segment: str) -> List[str]:
#         tokens: List[str] = []
#         buffer: List[str] = []
#         for idx, ch in enumerate(segment):
#             buffer.append(ch)
#             next_char = segment[idx + 1] if idx + 1 < len(segment) else ""
#             if ch.islower() and next_char.isupper():
#                 token = "".join(buffer).strip(" ,-:")
#                 if token:
#                     tokens.append(token)
#                 buffer = []
#         residual = "".join(buffer).strip(" ,-:")
#         if residual:
#             tokens.append(residual)
#         return tokens

#     raw_tokens = split_items(items_region)
#     item_names: List[str] = []
#     for token in raw_tokens:
#         cleaned = token.strip()
#         if not cleaned:
#             continue
#         lowered = cleaned.lower()
#         if any(
#             keyword in lowered
#             for keyword in ("floor plan", "amount", "division", "payment", "btw", "tax")
#         ):
#             continue
#         if lowered.startswith(("table", "taf")):
#             continue
#         item_names.append(cleaned)

#     price_pattern = re.compile(r"\d+[.,]\d{2}")
#     total_match = re.search(r"€\s*([\d.,]+)", text)
#     total_amount = float(total_match.group(1).replace(",", ".")) if total_match else None
#     total_idx = total_match.start() if total_match else len(text)

#     price_matches_zone = [
#         m
#         for m in price_pattern.finditer(text)
#         if m.start() >= items_region_start and m.start() < total_idx
#     ]
#     price_values_zone = [float(m.group().replace(",", ".")) for m in price_matches_zone]
#     if item_names and len(price_values_zone) >= len(item_names):
#         price_values = price_values_zone[-len(item_names):]
#     else:
#         price_values = price_values_zone

#     items: List[Dict] = []
#     for idx, name in enumerate(item_names):
#         price_value: Optional[float] = None
#         if idx < len(price_values):
#             price_value = round(price_values[idx], 2)
#             # Round to two decimals consistently
#             price_value = float(f"{price_value:.2f}")
#         items.append({"name": name, "price": price_value})

#     if total_amount is None and items:
#         summed = sum(item["price"] for item in items if item["price"] is not None)
#         total_amount = round(summed, 2) if summed else None

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

#     markdown_lines = [f"# {store_name}", ""]
#     if date or time:
#         markdown_lines.append(f"**Date:** {date} {time}".strip())
#     if table:
#         markdown_lines.append(f"**Table:** {table}")
#     if statement_number:
#         markdown_lines.append(f"**Statement:** {statement_number}")
#     assignment_parts = []
#     if order_device:
#         assignment_parts.append(f"Device: {order_device}")
#     if employee_no:
#         assignment_parts.append(f"Employee: {employee_no}")
#     if assignment_parts:
#         markdown_lines.append("**Assignment:** " + " | ".join(assignment_parts))
#     markdown_lines.append("\n---\n")
#     markdown_lines.append("| Item | Price (€) |\n|------|-----------:|")
#     for item in items:
#         price_text = f"{item['price']:.2f}" if item["price"] is not None else ""
#         markdown_lines.append(f"| {item['name']} | {price_text} |")
#     if total_amount is not None:
#         markdown_lines.append(f"| **Total** | **{total_amount:.2f}** |")
#     markdown_lines.append("")
#     if tax_id:
#         markdown_lines.append(f"**Tax ID:** {tax_id}")
#     markdown_text = "\n".join(markdown_lines)

#     return receipt_data, markdown_text

# import re
# from typing import List, Dict, Optional, Tuple


# def parse_receipt_text(full_text: str) -> Tuple[Dict, str]:
#     """
#     Convert OCR text into structured data (schema) + markdown view.
#     Assumes receipt layout is fixed (Fluffy Café receipts).
#     Fixed to properly extract multiple items with individual prices.
#     """
#     text = (full_text or "").replace("\r", "")
#     store_match = re.search(r"(Fluffy\s+Cafe-?Restaur[a-z]*)", text, re.IGNORECASE)
#     store_name = (
#         re.sub(r"\s+", " ", store_match.group(1)).strip()
#         if store_match
#         else "Fluffy Cafe-Restaurant"
#     )

#     terminal_match = re.search(r"([A-Za-z][A-Za-z0-9]*)\s*/\s*([A-Za-z0-9\-]+)", text)
#     order_device = terminal_match.group(1).strip() if terminal_match else None
#     employee_no = terminal_match.group(2).strip() if terminal_match else None

#     statement_match = re.search(r"Statement\s+([A-Z0-9.\-]+)", text, re.IGNORECASE)
#     statement_number = statement_match.group(1) if statement_match else None

#     datetime_match = re.search(r"(\d{2}/\d{2}/\d{4}),\s*(\d{2}:\d{2})", text)
#     date = datetime_match.group(1) if datetime_match else ""
#     time = datetime_match.group(2) if datetime_match else ""

#     table_match = re.search(r"(?:Tafel|Table)\s*(\d+)", text, re.IGNORECASE)
#     table = table_match.group(1) if table_match else None

#     # Find the section between table and "Amount due"
#     items_region_start = table_match.end() if table_match else 0
#     items_region = text[items_region_start:]
#     amount_match = re.search(r"Amount\s+due", items_region, re.IGNORECASE)
#     if not amount_match:
#         amount_match = re.search(r"Amount", items_region, re.IGNORECASE)
#     if amount_match:
#         items_region = items_region[:amount_match.start()]

#     # Extract all prices in this region
#     price_pattern = re.compile(r"(\d+[.,]\d{2})")
#     price_matches = list(price_pattern.finditer(items_region))
#     price_values = [float(m.group().replace(",", ".")) for m in price_matches]

#     # Split items_region by newlines and process line by line
#     lines = items_region.split('\n')
#     item_names: List[str] = []
    
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
        
#         # Remove leading/trailing special characters and whitespace
#         line = re.sub(r"^[\s•\-=:,]+|[\s•\-=:,]+$", "", line)
#         if not line:
#             continue
        
#         lowered = line.lower()
        
#         # Skip lines that are metadata/non-items
#         if any(
#             keyword in lowered
#             for keyword in ("floor plan", "amount", "division", "payment", "btw", "tax")
#         ):
#             continue
#         if lowered.startswith(("table", "taf", "restaurant")):
#             continue
        
#         # Remove any trailing price from the line itself
#         # (in case OCR put price on same line)
#         line_cleaned = re.sub(r'\s*\d+[.,]\d{2}\s*$', '', line).strip()
        
#         if line_cleaned:
#             item_names.append(line_cleaned)

#     # Match items with prices
#     # Use the last N prices for the N items found
#     items: List[Dict] = []
    
#     if item_names and len(price_values) >= len(item_names):
#         # Align: use the last N prices for N items
#         price_values = price_values[-len(item_names):]
#         for idx, name in enumerate(item_names):
#             price_value = round(price_values[idx], 2)
#             price_value = float(f"{price_value:.2f}")
#             items.append({"name": name, "price": price_value})
#     elif item_names:
#         # Fewer prices than items - assign what we have
#         for idx, name in enumerate(item_names):
#             price_value: Optional[float] = None
#             if idx < len(price_values):
#                 price_value = round(price_values[idx], 2)
#                 price_value = float(f"{price_value:.2f}")
#             items.append({"name": name, "price": price_value})

#     # Extract total amount
#     total_match = re.search(r"€\s*([\d.,]+)", text)
#     total_amount = float(total_match.group(1).replace(",", ".")) if total_match else None

#     if total_amount is None and items:
#         summed = sum(item["price"] for item in items if item["price"] is not None)
#         total_amount = round(summed, 2) if summed else None

#     # Extract tax ID
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

#     # Build markdown view
#     markdown_lines = [f"# {store_name}", ""]
#     if date or time:
#         markdown_lines.append(f"**Date:** {date} {time}".strip())
#     if table:
#         markdown_lines.append(f"**Table:** {table}")
#     if statement_number:
#         markdown_lines.append(f"**Statement:** {statement_number}")
#     assignment_parts = []
#     if order_device:
#         assignment_parts.append(f"Device: {order_device}")
#     if employee_no:
#         assignment_parts.append(f"Employee: {employee_no}")
#     if assignment_parts:
#         markdown_lines.append("**Assignment:** " + " | ".join(assignment_parts))
#     markdown_lines.append("\n---\n")
#     markdown_lines.append("| Item | Price (€) |\n|------|-----------:|")
#     for item in items:
#         price_text = f"{item['price']:.2f}" if item["price"] is not None else ""
#         markdown_lines.append(f"| {item['name']} | {price_text} |")
#     if total_amount is not None:
#         markdown_lines.append(f"| **Total** | **{total_amount:.2f}** |")
#     markdown_lines.append("")
#     if tax_id:
#         markdown_lines.append(f"**Tax ID:** {tax_id}")
#     markdown_text = "\n".join(markdown_lines)

#     return receipt_data, markdown_text

# import re
# from typing import List, Dict, Optional, Tuple


# def parse_receipt_text(full_text: str) -> Tuple[Dict, str]:
#     """
#     Convert OCR text into structured data (schema) + markdown view.
#     Assumes receipt layout is fixed (Fluffy Café receipts).
#     Fixed to properly extract multiple items with individual prices.
#     """
#     text = (full_text or "").replace("\r", "")
#     store_match = re.search(r"(Fluffy\s+Cafe-?Restaur[a-z]*)", text, re.IGNORECASE)
#     store_name = (
#         re.sub(r"\s+", " ", store_match.group(1)).strip()
#         if store_match
#         else "Fluffy Cafe-Restaurant"
#     )

#     terminal_match = re.search(r"([A-Za-z][A-Za-z0-9]*)\s*/\s*([A-Za-z0-9\-]+)", text)
#     order_device = terminal_match.group(1).strip() if terminal_match else None
#     employee_no = terminal_match.group(2).strip() if terminal_match else None

#     statement_match = re.search(r"Statement\s+([A-Z0-9.\-]+)", text, re.IGNORECASE)
#     statement_number = statement_match.group(1) if statement_match else None

#     datetime_match = re.search(r"(\d{2}/\d{2}/\d{4}),\s*(\d{2}:\d{2})", text)
#     date = datetime_match.group(1) if datetime_match else ""
#     time = datetime_match.group(2) if datetime_match else ""

#     table_match = re.search(r"(?:Tafel|Table)\s*(\d+)", text, re.IGNORECASE)
#     table = table_match.group(1) if table_match else None

#     # Find the section between table and "Amount due"
#     items_region_start = table_match.end() if table_match else 0
#     items_region = text[items_region_start:]
#     amount_match = re.search(r"Amount\s+due", items_region, re.IGNORECASE)
#     if not amount_match:
#         amount_match = re.search(r"Amount", items_region, re.IGNORECASE)
#     if amount_match:
#         items_region = items_region[:amount_match.start()]

#     # Extract all prices in this region BEFORE processing
#     price_pattern = re.compile(r"(\d+[.,]\d{2})")
#     price_matches = list(price_pattern.finditer(items_region))
#     price_values = [float(m.group().replace(",", ".")) for m in price_matches]
    
#     # Debug: ensure we found prices
#     if not price_values:
#         # Fallback: search in the entire text between table and total
#         total_match_temp = re.search(r"€\s*([\d.,]+)", text)
#         total_idx = total_match_temp.start() if total_match_temp else len(text)
#         fallback_region = text[items_region_start:total_idx]
#         price_matches = list(price_pattern.finditer(fallback_region))
#         price_values = [float(m.group().replace(",", ".")) for m in price_matches]

#     # Split items_region by newlines and process line by line
#     lines = items_region.split('\n')
#     item_names: List[str] = []
    
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
        
#         # Remove leading/trailing special characters and whitespace
#         line = re.sub(r"^[\s•\-=:,]+|[\s•\-=:,]+$", "", line)
#         if not line:
#             continue
        
#         lowered = line.lower()
        
#         # Skip lines that are metadata/non-items
#         if any(
#             keyword in lowered
#             for keyword in ("floor plan", "amount", "division", "payment", "btw", "tax")
#         ):
#             continue
#         if lowered.startswith(("table", "taf", "restaurant")):
#             continue
        
#         # Remove any trailing price from the line itself
#         # (in case OCR put price on same line)
#         line_cleaned = re.sub(r'\s*\d+[.,]\d{2}\s*$', '', line).strip()
        
#         if line_cleaned:
#             item_names.append(line_cleaned)

#     # Match items with prices
#     # Use the last N prices for the N items found
#     items: List[Dict] = []
    
#     if item_names and len(price_values) >= len(item_names):
#         # Align: use the last N prices for N items
#         price_values = price_values[-len(item_names):]
#         for idx, name in enumerate(item_names):
#             price_value = round(price_values[idx], 2)
#             price_value = float(f"{price_value:.2f}")
#             items.append({"name": name, "price": price_value})
#     elif item_names:
#         # Fewer prices than items - assign what we have
#         for idx, name in enumerate(item_names):
#             price_value: Optional[float] = None
#             if idx < len(price_values):
#                 price_value = round(price_values[idx], 2)
#                 price_value = float(f"{price_value:.2f}")
#             items.append({"name": name, "price": price_value})

#     # Extract total amount
#     total_match = re.search(r"€\s*([\d.,]+)", text)
#     total_amount = float(total_match.group(1).replace(",", ".")) if total_match else None

#     if total_amount is None and items:
#         summed = sum(item["price"] for item in items if item["price"] is not None)
#         total_amount = round(summed, 2) if summed else None

#     # Extract tax ID
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

#     # Build markdown view
#     markdown_lines = [f"# {store_name}", ""]
#     if date or time:
#         markdown_lines.append(f"**Date:** {date} {time}".strip())
#     if table:
#         markdown_lines.append(f"**Table:** {table}")
#     if statement_number:
#         markdown_lines.append(f"**Statement:** {statement_number}")
#     assignment_parts = []
#     if order_device:
#         assignment_parts.append(f"Device: {order_device}")
#     if employee_no:
#         assignment_parts.append(f"Employee: {employee_no}")
#     if assignment_parts:
#         markdown_lines.append("**Assignment:** " + " | ".join(assignment_parts))
#     markdown_lines.append("\n---\n")
#     markdown_lines.append("| Item | Price (€) |\n|------|-----------:|")
#     for item in items:
#         price_text = f"{item['price']:.2f}" if item["price"] is not None else ""
#         markdown_lines.append(f"| {item['name']} | {price_text} |")
#     if total_amount is not None:
#         markdown_lines.append(f"| **Total** | **{total_amount:.2f}** |")
#     markdown_lines.append("")
#     if tax_id:
#         markdown_lines.append(f"**Tax ID:** {tax_id}")
#     markdown_text = "\n".join(markdown_lines)

#     return receipt_data, markdown_text

# import re
# from typing import List, Dict, Optional, Tuple


# def parse_receipt_text(full_text: str) -> Tuple[Dict, str]:
#     """
#     Convert OCR text into structured data (schema) + markdown view.
#     Assumes receipt layout is fixed (Fluffy Café receipts).
#     Fixed to properly extract multiple items with individual prices.
#     """
#     text = (full_text or "").replace("\r", "")
#     store_match = re.search(r"(Fluffy\s+Cafe-?Restaur[a-z]*)", text, re.IGNORECASE)
#     store_name = (
#         re.sub(r"\s+", " ", store_match.group(1)).strip()
#         if store_match
#         else "Fluffy Cafe-Restaurant"
#     )

#     terminal_match = re.search(r"([A-Za-z][A-Za-z0-9]*)\s*/\s*([A-Za-z0-9\-]+)", text)
#     order_device = terminal_match.group(1).strip() if terminal_match else None
#     employee_no = terminal_match.group(2).strip() if terminal_match else None

#     # Extract reference_no (appears on same line as terminal/employee info, on the right side)
#     reference_match = re.search(r"([A-Za-z][A-Za-z0-9]*)\s*/\s*([A-Za-z0-9\-]+)\s+([A-Z]\d+[A-Z0-9.]*)", text)
#     reference_no = reference_match.group(3).strip() if reference_match else None

#     statement_match = re.search(r"Statement\s+([A-Z0-9.\-]+)", text, re.IGNORECASE)
#     statement_number = statement_match.group(1) if statement_match else None

#     datetime_match = re.search(r"(\d{2}/\d{2}/\d{4}),\s*(\d{2}:\d{2})", text)
#     date = datetime_match.group(1) if datetime_match else ""
#     time = datetime_match.group(2) if datetime_match else ""

#     table_match = re.search(r"(?:Tafel|Table)\s*(\d+)", text, re.IGNORECASE)
#     table = table_match.group(1) if table_match else None

#     # Find the section between table and "Amount due"
#     items_region_start = table_match.end() if table_match else 0
#     items_region = text[items_region_start:]
#     amount_match = re.search(r"Amount\s+due", items_region, re.IGNORECASE)
#     if not amount_match:
#         amount_match = re.search(r"Amount", items_region, re.IGNORECASE)
#     if amount_match:
#         items_region = items_region[:amount_match.start()]

#     # Extract all prices in this region BEFORE processing
#     price_pattern = re.compile(r"(\d+[.,]\d{2})")
#     price_matches = list(price_pattern.finditer(items_region))
#     price_values = [float(m.group().replace(",", ".")) for m in price_matches]
    
#     # Debug: ensure we found prices
#     if not price_values:
#         # Fallback: search in the entire text between table and total
#         total_match_temp = re.search(r"€\s*([\d.,]+)", text)
#         total_idx = total_match_temp.start() if total_match_temp else len(text)
#         fallback_region = text[items_region_start:total_idx]
#         price_matches = list(price_pattern.finditer(fallback_region))
#         price_values = [float(m.group().replace(",", ".")) for m in price_matches]

#     # Split items_region by newlines and process line by line
#     lines = items_region.split('\n')
#     item_names: List[str] = []
    
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
        
#         # Remove leading/trailing special characters and whitespace
#         line = re.sub(r"^[\s•\-=:,]+|[\s•\-=:,]+$", "", line)
#         if not line:
#             continue
        
#         lowered = line.lower()
        
#         # Skip lines that are metadata/non-items
#         if any(
#             keyword in lowered
#             for keyword in ("floor plan", "amount", "division", "payment", "btw", "tax")
#         ):
#             continue
#         if lowered.startswith(("table", "taf", "restaurant")):
#             continue
        
#         # Remove any trailing price from the line itself
#         # (in case OCR put price on same line)
#         line_cleaned = re.sub(r'\s*\d+[.,]\d{2}\s*$', '', line).strip()
        
#         if line_cleaned:
#             item_names.append(line_cleaned)

#     # Match items with prices
#     # Use the last N prices for the N items found
#     items: List[Dict] = []
    
#     if item_names and len(price_values) >= len(item_names):
#         # Align: use the last N prices for N items
#         price_values = price_values[-len(item_names):]
#         for idx, name in enumerate(item_names):
#             price_value = round(price_values[idx], 2)
#             price_value = float(f"{price_value:.2f}")
#             items.append({"name": name, "price": price_value})
#     elif item_names:
#         # Fewer prices than items - assign what we have
#         for idx, name in enumerate(item_names):
#             price_value: Optional[float] = None
#             if idx < len(price_values):
#                 price_value = round(price_values[idx], 2)
#                 price_value = float(f"{price_value:.2f}")
#             items.append({"name": name, "price": price_value})

#     # Extract total amount
#     total_match = re.search(r"€\s*([\d.,]+)", text)
#     total_amount = float(total_match.group(1).replace(",", ".")) if total_match else None

#     if total_amount is None and items:
#         summed = sum(item["price"] for item in items if item["price"] is not None)
#         total_amount = round(summed, 2) if summed else None

#     # Extract tax ID
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

#     # Build markdown view
#     markdown_lines = [f"# {store_name}", ""]
#     if date or time:
#         markdown_lines.append(f"**Date:** {date} {time}".strip())
#     if table:
#         markdown_lines.append(f"**Table:** {table}")
#     if statement_number:
#         markdown_lines.append(f"**Statement:** {statement_number}")
#     assignment_parts = []
#     if order_device:
#         assignment_parts.append(f"Device: {order_device}")
#     if employee_no:
#         assignment_parts.append(f"Employee: {employee_no}")
#     if assignment_parts:
#         markdown_lines.append("**Assignment:** " + " | ".join(assignment_parts))
#     markdown_lines.append("\n---\n")
#     markdown_lines.append("| Item | Price (€) |\n|------|-----------:|")
#     for item in items:
#         price_text = f"{item['price']:.2f}" if item["price"] is not None else ""
#         markdown_lines.append(f"| {item['name']} | {price_text} |")
#     if total_amount is not None:
#         markdown_lines.append(f"| **Total** | **{total_amount:.2f}** |")
#     markdown_lines.append("")
#     if tax_id:
#         markdown_lines.append(f"**Tax ID:** {tax_id}")
#     markdown_text = "\n".join(markdown_lines)

#     return receipt_data, markdown_text

# import re
# from typing import List, Dict, Optional, Tuple


# def parse_receipt_text(full_text: str) -> Tuple[Dict, str]:
#     """
#     Convert OCR text into structured data (schema) + markdown view.
#     Assumes receipt layout is fixed (Fluffy Café receipts).
#     Fixed to properly extract multiple items with individual prices.
#     """
#     text = (full_text or "").replace("\r", "")
#     store_match = re.search(r"(Fluffy\s+Cafe-?Restaur[a-z]*)", text, re.IGNORECASE)
#     store_name = (
#         re.sub(r"\s+", " ", store_match.group(1)).strip()
#         if store_match
#         else "Fluffy Cafe-Restaurant"
#     )

#     terminal_match = re.search(r"([A-Za-z][A-Za-z0-9]*)\s*/\s*([A-Za-z0-9\-]+)", text)
#     order_device = terminal_match.group(1).strip() if terminal_match else None
#     employee_no = terminal_match.group(2).strip() if terminal_match else None

#     reference_match = re.search(r"([A-Za-z][A-Za-z0-9]*)\s*/\s*([A-Za-z0-9\-]+)\s+([A-Z]\d+[A-Z0-9.]*)", text)
#     print(f"reference_match: {reference_match}")
#     reference_no = reference_match.group(3).strip() if reference_match else None
#     print(f"reference_no: {reference_no}")
    
#     statement_match = re.search(r"Statement\s+([A-Z0-9.\-]+)", text, re.IGNORECASE)
#     statement_number = statement_match.group(1) if statement_match else None

#     datetime_match = re.search(r"(\d{2}/\d{2}/\d{4}),\s*(\d{2}:\d{2})", text)
#     date = datetime_match.group(1) if datetime_match else ""
#     time = datetime_match.group(2) if datetime_match else ""

#     table_match = re.search(r"(?:Tafel|Table)\s*(\d+)", text, re.IGNORECASE)
#     table = table_match.group(1) if table_match else None

#     # Find the section between table and "Amount due"
#     items_region_start = table_match.end() if table_match else 0
#     items_region = text[items_region_start:]
#     amount_match = re.search(r"Amount\s+due", items_region, re.IGNORECASE)
#     if not amount_match:
#         amount_match = re.search(r"Amount", items_region, re.IGNORECASE)
#     if amount_match:
#         items_region = items_region[:amount_match.start()]

#     # Extract all prices in this region BEFORE processing
#     price_pattern = re.compile(r"(\d+[.,]\d{2})")
#     price_matches = list(price_pattern.finditer(items_region))
#     price_values = [float(m.group().replace(",", ".")) for m in price_matches]
    
#     # Debug: ensure we found prices
#     if not price_values:
#         # Fallback: search in the entire text between table and total
#         total_match_temp = re.search(r"€\s*([\d.,]+)", text)
#         total_idx = total_match_temp.start() if total_match_temp else len(text)
#         fallback_region = text[items_region_start:total_idx]
#         price_matches = list(price_pattern.finditer(fallback_region))
#         price_values = [float(m.group().replace(",", ".")) for m in price_matches]

#     # Split items_region by newlines and process line by line
#     lines = items_region.split('\n')
#     item_names: List[str] = []
    
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
        
#         # Remove leading/trailing special characters and whitespace
#         line = re.sub(r"^[\s•\-=:,]+|[\s•\-=:,]+$", "", line)
#         if not line:
#             continue
        
#         lowered = line.lower()
        
#         # Skip lines that are metadata/non-items
#         if any(
#             keyword in lowered
#             for keyword in ("floor plan", "amount", "division", "payment", "btw", "tax")
#         ):
#             continue
#         if lowered.startswith(("table", "taf", "restaurant")):
#             continue
        
#         # Remove any trailing price from the line itself
#         # (in case OCR put price on same line)
#         line_cleaned = re.sub(r'\s*\d+[.,]\d{2}\s*$', '', line).strip()
        
#         if line_cleaned:
#             item_names.append(line_cleaned)

#     # Match items with prices
#     # Use the last N prices for the N items found
#     items: List[Dict] = []
    
#     if item_names and len(price_values) >= len(item_names):
#         # Align: use the last N prices for N items
#         price_values = price_values[-len(item_names):]
#         for idx, name in enumerate(item_names):
#             price_value = round(price_values[idx], 2)
#             price_value = float(f"{price_value:.2f}")
#             items.append({"name": name, "price": price_value})
#     elif item_names:
#         # Fewer prices than items - assign what we have
#         for idx, name in enumerate(item_names):
#             price_value: Optional[float] = None
#             if idx < len(price_values):
#                 price_value = round(price_values[idx], 2)
#                 price_value = float(f"{price_value:.2f}")
#             items.append({"name": name, "price": price_value})

#     # Extract total amount
#     total_match = re.search(r"€\s*([\d.,]+)", text)
#     total_amount = float(total_match.group(1).replace(",", ".")) if total_match else None

#     if total_amount is None and items:
#         summed = sum(item["price"] for item in items if item["price"] is not None)
#         total_amount = round(summed, 2) if summed else None

#     # Extract tax ID
#     tax_match = re.search(r"Btw[: ]+([A-Z0-9]+)", text, re.IGNORECASE)
#     tax_id = tax_match.group(1) if tax_match else None

#     receipt_data = {
#         "store_name": store_name,
#         "statement_number": statement_number,
#         "order_device": order_device,
#         "employee_no": employee_no,
#         "reference_no": reference_no,
#         "date": date,
#         "time": time,
#         "table": table,
#         "items": items,
#         "total_amount": total_amount,
#         "tax_id": tax_id,
#     }

#     # Build markdown view
#     markdown_lines = [f"# {store_name}", ""]
#     if date or time:
#         markdown_lines.append(f"**Date:** {date} {time}".strip())
#     if table:
#         markdown_lines.append(f"**Table:** {table}")
#     if statement_number:
#         markdown_lines.append(f"**Statement:** {statement_number}")
#     assignment_parts = []
#     if order_device:
#         assignment_parts.append(f"Device: {order_device}")
#     if employee_no:
#         assignment_parts.append(f"Employee: {employee_no}")
#     if assignment_parts:
#         markdown_lines.append("**Assignment:** " + " | ".join(assignment_parts))
#     markdown_lines.append("\n---\n")
#     markdown_lines.append("| Item | Price (€) |\n|------|-----------:|")
#     for item in items:
#         price_text = f"{item['price']:.2f}" if item["price"] is not None else ""
#         markdown_lines.append(f"| {item['name']} | {price_text} |")
#     if total_amount is not None:
#         markdown_lines.append(f"| **Total** | **{total_amount:.2f}** |")
#     markdown_lines.append("")
#     if tax_id:
#         markdown_lines.append(f"**Tax ID:** {tax_id}")
#     markdown_text = "\n".join(markdown_lines)

#     return receipt_data, markdown_text

# import re
# from typing import List, Dict, Optional, Tuple


# def parse_receipt_text(full_text: str) -> Tuple[Dict, str]:
#     """
#     Convert OCR text into structured data (schema) + markdown view.
#     Assumes receipt layout is fixed (Fluffy Café receipts).
#     Fixed to properly extract multiple items with individual prices.
#     """
#     text = (full_text or "").replace("\r", "")
#     store_match = re.search(r"(Fluffy\s+Cafe-?Restaur[a-z]*)", text, re.IGNORECASE)
#     store_name = (
#         re.sub(r"\s+", " ", store_match.group(1)).strip()
#         if store_match
#         else "Fluffy Cafe-Restaurant"
#     )

#     terminal_match = re.search(r"([A-Za-z][A-Za-z0-9]*)\s*/\s*([A-Za-z0-9\-]+)", text)
#     order_device = terminal_match.group(1).strip() if terminal_match else None
#     employee_no = terminal_match.group(2).strip() if terminal_match else None

#     # Extract reference_no - it appears after Statement, on its own or on same line
#     # Look for pattern: A followed by digits, dots, etc.
#     reference_match = re.search(r"[Ss]tatement\s+[A-Z0-9.\-]+\s+([A-Z]\d+[A-Z0-9.]*)", text)
#     reference_no = reference_match.group(1).strip() if reference_match else None

#     statement_match = re.search(r"Statement\s+([A-Z0-9.\-]+)", text, re.IGNORECASE)
#     statement_number = statement_match.group(1) if statement_match else None

#     datetime_match = re.search(r"(\d{2}/\d{2}/\d{4}),\s*(\d{2}:\d{2})", text)
#     date = datetime_match.group(1) if datetime_match else ""
#     time = datetime_match.group(2) if datetime_match else ""

#     table_match = re.search(r"(?:Tafel|Table)\s*(\d+)", text, re.IGNORECASE)
#     table = table_match.group(1) if table_match else None

#     # Extract total amount first - this marks the end of items section
#     total_match = re.search(r"€\s*([\d.,]+)", text)
#     total_amount = float(total_match.group(1).replace(",", ".")) if total_match else None
#     total_idx = total_match.start() if total_match else len(text)

#     # Find the section between table and € symbol (which marks total)
#     items_region_start = table_match.end() if table_match else 0
#     items_region = text[items_region_start:total_idx]

#     # Extract all prices in this region BEFORE processing
#     price_pattern = re.compile(r"(\d+[.,]\d{2})")
#     price_matches = list(price_pattern.finditer(items_region))
#     price_values = [float(m.group().replace(",", ".")) for m in price_matches]

#     # Split items_region by newlines and process line by line
#     lines = items_region.split('\n')
#     item_names: List[str] = []
    
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
        
#         # Remove leading/trailing special characters and whitespace
#         line = re.sub(r"^[\s•\-=:,]+|[\s•\-=:,]+$", "", line)
#         if not line:
#             continue
        
#         # Skip if line is just the euro symbol or contains it
#         if line == "€" or line.startswith("€"):
#             continue
        
#         lowered = line.lower()
        
#         # Skip lines that are metadata/non-items
#         if any(
#             keyword in lowered
#             for keyword in ("floor plan", "amount", "division", "payment", "btw", "tax", "dank voor", "lightspeed")
#         ):
#             continue
#         if lowered.startswith(("table", "taf", "restaurant")):
#             continue
        
#         # Remove any trailing price from the line itself
#         # (in case OCR put price on same line)
#         line_cleaned = re.sub(r'\s*\d+[.,]\d{2}\s*$', '', line).strip()
        
#         if line_cleaned:
#             item_names.append(line_cleaned)

#     # Match items with prices
#     # Use the last N prices for the N items found
#     items: List[Dict] = []
    
#     if item_names and len(price_values) >= len(item_names):
#         # Align: use the last N prices for N items
#         price_values = price_values[-len(item_names):]
#         for idx, name in enumerate(item_names):
#             price_value = round(price_values[idx], 2)
#             price_value = float(f"{price_value:.2f}")
#             items.append({"name": name, "price": price_value})
#     elif item_names:
#         # Fewer prices than items - assign what we have
#         for idx, name in enumerate(item_names):
#             price_value: Optional[float] = None
#             if idx < len(price_values):
#                 price_value = round(price_values[idx], 2)
#                 price_value = float(f"{price_value:.2f}")
#             items.append({"name": name, "price": price_value})

#     if total_amount is None and items:
#         summed = sum(item["price"] for item in items if item["price"] is not None)
#         total_amount = round(summed, 2) if summed else None

#     # Extract tax ID
#     tax_match = re.search(r"Btw[: ]+([A-Z0-9]+)", text, re.IGNORECASE)
#     tax_id = tax_match.group(1) if tax_match else None

#     receipt_data = {
#         "store_name": store_name,
#         "statement_number": statement_number,
#         "order_device": order_device,
#         "employee_no": employee_no,
#         "reference_no": reference_no,
#         "date": date,
#         "time": time,
#         "table": table,
#         "items": items,
#         "total_amount": total_amount,
#         "tax_id": tax_id,
#     }

#     # Build markdown view
#     markdown_lines = [f"# {store_name}", ""]
#     if date or time:
#         markdown_lines.append(f"**Date:** {date} {time}".strip())
#     if table:
#         markdown_lines.append(f"**Table:** {table}")
#     if statement_number:
#         markdown_lines.append(f"**Statement:** {statement_number}")
#     if reference_no:
#         markdown_lines.append(f"**Reference:** {reference_no}")
#     assignment_parts = []
#     if order_device:
#         assignment_parts.append(f"Device: {order_device}")
#     if employee_no:
#         assignment_parts.append(f"Employee: {employee_no}")
#     if assignment_parts:
#         markdown_lines.append("**Assignment:** " + " | ".join(assignment_parts))
#     markdown_lines.append("\n---\n")
#     markdown_lines.append("| Item | Price (€) |\n|------|-----------:|")
#     for item in items:
#         price_text = f"{item['price']:.2f}" if item["price"] is not None else ""
#         markdown_lines.append(f"| {item['name']} | {price_text} |")
#     if total_amount is not None:
#         markdown_lines.append(f"| **Total** | **{total_amount:.2f}** |")
#     markdown_lines.append("")
#     if tax_id:
#         markdown_lines.append(f"**Tax ID:** {tax_id}")
#     markdown_text = "\n".join(markdown_lines)

#     return receipt_data, markdown_text

import re
from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class OCRToken:
    text: str
    x: float
    y: float
    width: float
    height: float

# def parse_receipt_text(full_text: str, response=None) -> Tuple[Dict, str]:
#     """
#     Convert OCR text into structured data (schema) + markdown view.
#     Requires the Vision `document_text_detection` response so we can use bounding boxes.
#     """
#     if response is None or not response.full_text_annotation.pages:
#         raise ValueError("Vision response with layout required for stable parsing.")

#     page = response.full_text_annotation.pages[0]

#     def token_stream():
#         for block in page.blocks:
#             for paragraph in block.paragraphs:
#                 for word in paragraph.words:
#                     word_text = "".join(symbol.text for symbol in word.symbols)
#                     vertices = word.bounding_box.vertices
#                     xs = [v.x for v in vertices]
#                     ys = [v.y for v in vertices]
#                     x_min, x_max = min(xs), max(xs)
#                     y_min, y_max = min(ys), max(ys)
#                     yield OCRToken(
#                         text=word_text,
#                         x=x_min,
#                         y=y_min,
#                         width=x_max - x_min,
#                         height=y_max - y_min,
#                     )

#     tokens: List[OCRToken] = list(token_stream())
#     tokens.sort(key=lambda t: (round(t.y / 6), t.x))  # cluster by approximate line height

#     # 1. Recover metadata using the safest fields from `full_text`.
#     text = full_text.replace("\r", "")
#     store_match = re.search(r"(Fluffy\s+Cafe-?Restaur[a-z]*)", text, re.IGNORECASE)
#     store_name = (
#         re.sub(r"\s+", " ", store_match.group(1)).strip()
#         if store_match
#         else "Fluffy Cafe-Restaurant"
#     )
#     terminal_match = re.search(r"([A-Za-z][A-Za-z0-9]*)\s*/\s*([A-Za-z0-9\-]+)", text)
#     order_device = terminal_match.group(1) if terminal_match else None
#     employee_no = terminal_match.group(2) if terminal_match else None
#     statement_match = re.search(r"Statement\s+([A-Z0-9.\-]+)", text, re.IGNORECASE)
#     statement_number = statement_match.group(1) if statement_match else None
#     date_match = re.search(r"(\d{2}/\d{2}/\d{4})", text)
#     time_match = re.search(r"\d{2}:\d{2}", text)
#     date = date_match.group(1) if date_match else ""
#     time = time_match.group(0) if time_match else ""
#     table_match = re.search(r"(?:Tafel|Table)\s*(\d+)", text, re.IGNORECASE)
#     table = table_match.group(1) if table_match else None
#     total_match = re.search(r"€\s*([\d.,]+)", text)
#     total_amount = float(total_match.group(1).replace(",", ".")) if total_match else None
#     tax_match = re.search(r"Btw[: ]+([A-Z0-9]+)", text, re.IGNORECASE)
#     tax_id = tax_match.group(1) if tax_match else None

#     # 2. Determine horizontal split (names vs prices) by median X of euro prices.
#     price_tokens = [t for t in tokens if re.fullmatch(r"\d+[.,]\d{2}", t.text)]
#     if not price_tokens:
#         raise ValueError("No price tokens detected in layout; cannot build item list.")

#     price_column_x = statistics.median(t.x for t in price_tokens)
#     row_clusters: List[List[OCRToken]] = []

#     for token in tokens:
#         assigned = False
#         for row in row_clusters:
#             if abs(row[0].y - token.y) <= max(row[0].height, token.height) * 0.6:
#                 row.append(token)
#                 assigned = True
#                 break
#         if not assigned:
#             row_clusters.append([token])

#     items: List[Dict[str, Optional[float]]] = []
#     for row in row_clusters:
#         row.sort(key=lambda t: t.x)
#         left_words = [t.text for t in row if t.x + t.width < price_column_x]
#         right_prices = [t.text for t in row if t.x >= price_column_x or t.text == "€"]

#         if not left_words:
#             continue

#         possible_price = None
#         for candidate in reversed(right_prices):
#             if re.fullmatch(r"\d+[.,]\d{2}", candidate):
#                 possible_price = float(candidate.replace(",", "."))
#                 break

#         name = " ".join(left_words).strip(" :-")
#         if any(k in name.lower() for k in ("floor plan", "amount due", "division hint")):
#             continue

#         items.append({"name": name, "price": possible_price})

#     # 3. Keep only meaningful rows: at least two characters, price or subsequent price.
#     filtered: List[Dict[str, Optional[float]]] = []
#     for item in items:
#         if len(item["name"]) < 2:
#             continue
#         filtered.append(item)
#     items = filtered

#     # 4. Fall back: if a row has no price, look for the next price-only row.
#     for idx, item in enumerate(items):
#         if item["price"] is None:
#             for future in items[idx + 1:]:
#                 if future["price"] is not None and future["name"] == "":
#                     item["price"] = future["price"]
#                     future["price"] = None
#                     break

#     # 5. Remove empty placeholder rows produced by previous step.
#     items = [item for item in items if item["name"]]

#     # 6. Recompute total if missing.
#     if total_amount is None and items:
#         prices = [item["price"] for item in items if item["price"] is not None]
#         total_amount = round(sum(prices), 2) if prices else None

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

#     markdown_lines = [f"# {store_name}", ""]
#     if date or time:
#         markdown_lines.append(f"**Date:** {date} {time}".strip())
#     if table:
#         markdown_lines.append(f"**Table:** {table}")
#     if statement_number:
#         markdown_lines.append(f"**Statement:** {statement_number}")
#     assignment_parts = []
#     if order_device:
#         assignment_parts.append(f"Device: {order_device}")
#     if employee_no:
#         assignment_parts.append(f"Employee: {employee_no}")
#     if assignment_parts:
#         markdown_lines.append("**Assignment:** " + " | ".join(assignment_parts))
#     markdown_lines.append("\n---\n")
#     markdown_lines.append("| Item | Price (€) |\n|------|-----------:|")
#     for item in items:
#         price_text = f"{item['price']:.2f}" if item["price"] is not None else ""
#         markdown_lines.append(f"| {item['name']} | {price_text} |")
#     if total_amount is not None:
#         markdown_lines.append(f"| **Total** | **{total_amount:.2f}** |")
#     markdown_lines.append("")
#     if tax_id:
#         markdown_lines.append(f"**Tax ID:** {tax_id}")

#     markdown_text = "\n".join(markdown_lines)
#     return receipt_data, markdown_text

# def parse_receipt_text(full_text: str, response=None) -> Tuple[Dict, str]:
#     """
#     Convert OCR text into structured data (schema) + markdown view.
#     Requires the Vision `document_text_detection` response so we can use bounding boxes.
#     """
#     if response is None or not response.full_text_annotation.pages:
#         raise ValueError("Vision response with layout required for stable parsing.")

#     page = response.full_text_annotation.pages[0]

#     def token_stream():
#         for block in page.blocks:
#             for paragraph in block.paragraphs:
#                 for word in paragraph.words:
#                     word_text = "".join(symbol.text for symbol in word.symbols)
#                     vertices = word.bounding_box.vertices
#                     xs = [v.x for v in vertices]
#                     ys = [v.y for v in vertices]
#                     x_min, x_max = min(xs), max(xs)
#                     y_min, y_max = min(ys), max(ys)
#                     yield OCRToken(
#                         text=word_text,
#                         x=x_min,
#                         y=y_min,
#                         width=x_max - x_min,
#                         height=y_max - y_min,
#                     )

#     tokens: List[OCRToken] = list(token_stream())
#     tokens.sort(key=lambda t: (round(t.y / 6), t.x))

#     table_baseline: Optional[float] = None
#     for token in tokens:
#         if token.text.lower() in {"tafel", "table"}:
#             table_baseline = token.y + token.height / 2
#             break

#     price_tokens = [t for t in tokens if re.fullmatch(r"\d+[.,]\d{2}", t.text)]
#     if not price_tokens:
#         raise ValueError("No price tokens detected in layout; cannot build item list.")

#     price_column_x = statistics.median(t.x for t in price_tokens)
#     max_price_center_y = max(t.y + t.height / 2 for t in price_tokens)

#     row_clusters: List[List[OCRToken]] = []
#     for token in tokens:
#         placed = False
#         for cluster in row_clusters:
#             anchor = cluster[0]
#             anchor_center = anchor.y + anchor.height / 2
#             token_center = token.y + token.height / 2
#             if abs(anchor_center - token_center) <= max(anchor.height, token.height) * 0.6:
#                 cluster.append(token)
#                 placed = True
#                 break
#         if not placed:
#             row_clusters.append([token])

#     rows: List[Dict[str, Any]] = []
#     for cluster in row_clusters:
#         cluster.sort(key=lambda t: t.x)
#         center_y = statistics.mean(t.y + t.height / 2 for t in cluster)
#         rows.append({"tokens": cluster, "center_y": center_y})

#     text = full_text.replace("\r", "")
#     store_match = re.search(r"(Fluffy\s+Cafe-?Restaur[a-z]*)", text, re.IGNORECASE)
#     store_name = (
#         re.sub(r"\s+", " ", store_match.group(1)).strip()
#         if store_match
#         else "Fluffy Cafe-Restaurant"
#     )
#     terminal_match = re.search(r"([A-Za-z][A-Za-z0-9]*)\s*/\s*([A-Za-z0-9\-]+)", text)
#     order_device = terminal_match.group(1) if terminal_match else None
#     employee_no = terminal_match.group(2) if terminal_match else None
#     statement_match = re.search(r"Statement\s+([A-Z0-9.\-]+)", text, re.IGNORECASE)
#     statement_number = statement_match.group(1) if statement_match else None
#     date_match = re.search(r"(\d{2}/\d{2}/\d{4})", text)
#     time_match = re.search(r"\d{2}:\d{2}", text)
#     date = date_match.group(1) if date_match else ""
#     time = time_match.group(0) if time_match else ""
#     table_match = re.search(r"(?:Tafel|Table)\s*(\d+)", text, re.IGNORECASE)
#     table = table_match.group(1) if table_match else None
#     total_match = re.search(r"€\s*([\d.,]+)", text)
#     total_amount = float(total_match.group(1).replace(",", ".")) if total_match else None
#     tax_match = re.search(r"Btw[: ]+([A-Z0-9]+)", text, re.IGNORECASE)
#     tax_id = tax_match.group(1) if tax_match else None

#     non_item_keywords = {
#         "amount",
#         "draft",
#         "receipt",
#         "floor plan",
#         "division",
#         "payment",
#         "btw",
#         "tax",
#         "dank",
#         "lightspeed",
#         "amstelveen",
#         "statement",
#         "device",
#         "employee",
#         "table",
#         "taf",
#         "thank",
#         "visit",
#         "no payment",
#         "hint",
#         "nl",
#     }

#     items: List[Dict[str, Optional[float]]] = []
#     for row in rows:
#         center_y = row["center_y"]
#         if table_baseline is not None and center_y <= table_baseline:
#             continue
#         if center_y >= max_price_center_y + 18:
#             continue

#         row_tokens = row["tokens"]
#         cutoff = price_column_x - 4
#         left_words = [t.text for t in row_tokens if t.x + t.width <= cutoff]
#         if not left_words:
#             continue

#         possible_price = None
#         for candidate in reversed(row_tokens):
#             if candidate.x >= price_column_x - 6 and re.fullmatch(r"\d+[.,]\d{2}", candidate.text):
#                 possible_price = float(candidate.text.replace(",", "."))
#                 break
#         if possible_price is None:
#             continue

#         name = " ".join(left_words).strip(" :-")
#         name_lower = name.lower()
#         if len(name) < 2:
#             continue
#         if not re.search(r"[a-z]", name_lower):
#             continue
#         if any(keyword in name_lower for keyword in non_item_keywords):
#             continue

#         items.append({"name": name, "price": round(possible_price, 2)})

#     if total_amount is None and items:
#         prices = [item["price"] for item in items if item["price"] is not None]
#         total_amount = round(sum(prices), 2) if prices else None

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

#     markdown_lines = [f"# {store_name}", ""]
#     if date or time:
#         markdown_lines.append(f"**Date:** {date} {time}".strip())
#     if table:
#         markdown_lines.append(f"**Table:** {table}")
#     if statement_number:
#         markdown_lines.append(f"**Statement:** {statement_number}")
#     assignment_parts = []
#     if order_device:
#         assignment_parts.append(f"Device: {order_device}")
#     if employee_no:
#         assignment_parts.append(f"Employee: {employee_no}")
#     if assignment_parts:
#         markdown_lines.append("**Assignment:** " + " | ".join(assignment_parts))
#     markdown_lines.append("\n---\n")
#     markdown_lines.append("| Item | Price (€) |\n|------|-----------:|")
#     for item in items:
#         price_text = f"{item['price']:.2f}" if item["price"] is not None else ""
#         markdown_lines.append(f"| {item['name']} | {price_text} |")
#     if total_amount is not None:
#         markdown_lines.append(f"| **Total** | **{total_amount:.2f}** |")
#     markdown_lines.append("")
#     if tax_id:
#         markdown_lines.append(f"**Tax ID:** {tax_id}")
#     markdown_text = "\n".join(markdown_lines)

#     return receipt_data, markdown_text


# def parse_receipt_text(full_text: str, response=None) -> Tuple[Dict, str]:
#     """
#     Convert OCR text into structured data (schema) + markdown view.
#     Requires the Vision `document_text_detection` response so we can use bounding boxes.
#     """
#     if response is None or not response.full_text_annotation.pages:
#         raise ValueError("Vision response with layout required for stable parsing.")

#     page = response.full_text_annotation.pages[0]

#     def token_stream():
#         for block in page.blocks:
#             for paragraph in block.paragraphs:
#                 for word in paragraph.words:
#                     word_text = "".join(symbol.text for symbol in word.symbols)
#                     vertices = word.bounding_box.vertices
#                     xs = [v.x for v in vertices]
#                     ys = [v.y for v in vertices]
#                     x_min, x_max = min(xs), max(xs)
#                     y_min, y_max = min(ys), max(ys)
#                     yield OCRToken(
#                         text=word_text,
#                         x=x_min,
#                         y=y_min,
#                         width=x_max - x_min,
#                         height=y_max - y_min,
#                     )

#     tokens: List[OCRToken] = list(token_stream())
#     tokens.sort(key=lambda t: (round(t.y / 6), t.x))

#     table_baseline: Optional[float] = None
#     for token in tokens:
#         if token.text.lower() in {"tafel", "table"}:
#             table_baseline = token.y + token.height / 2
#             break

#     price_tokens = [t for t in tokens if re.fullmatch(r"\d+[.,]\d{2}", t.text)]
#     if not price_tokens:
#         raise ValueError("No price tokens detected in layout; cannot build item list.")

#     price_column_x = statistics.median(t.x for t in price_tokens)
#     max_price_center_y = max(t.y + t.height / 2 for t in price_tokens)

#     row_clusters: List[List[OCRToken]] = []
#     for token in tokens:
#         placed = False
#         for cluster in row_clusters:
#             anchor = cluster[0]
#             anchor_center = anchor.y + anchor.height / 2
#             token_center = token.y + token.height / 2
#             if abs(anchor_center - token_center) <= max(anchor.height, token.height) * 0.6:
#                 cluster.append(token)
#                 placed = True
#                 break
#         if not placed:
#             row_clusters.append([token])

#     rows: List[Dict[str, Any]] = []
#     for cluster in row_clusters:
#         cluster.sort(key=lambda t: t.x)
#         center_y = statistics.mean(t.y + t.height / 2 for t in cluster)
#         rows.append({"tokens": cluster, "center_y": center_y})

#     text = full_text.replace("\r", "")
#     store_match = re.search(r"(Fluffy\s+Cafe-?Restaur[a-z]*)", text, re.IGNORECASE)
#     store_name = (
#         re.sub(r"\s+", " ", store_match.group(1)).strip()
#         if store_match
#         else "Fluffy Cafe-Restaurant"
#     )
#     terminal_match = re.search(r"([A-Za-z][A-Za-z0-9]*)\s*/\s*([A-Za-z0-9\-]+)", text)
#     order_device = terminal_match.group(1) if terminal_match else None
#     employee_no = terminal_match.group(2) if terminal_match else None

#     statement_no: Optional[str] = None
#     reference_no: Optional[str] = None
#     for idx, token in enumerate(tokens):
#         if token.text.lower() == "statement":
#             if idx + 1 < len(tokens):
#                 candidate_statement = tokens[idx + 1].text.strip(":")
#                 if re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]*", candidate_statement):
#                     statement_no = candidate_statement
#             if idx + 2 < len(tokens):
#                 candidate_reference = tokens[idx + 2].text.strip(":")
#                 if (
#                     re.fullmatch(r"[A-Z][A-Z0-9.\-]*", candidate_reference)
#                     and "/" not in candidate_reference
#                 ):
#                     reference_no = candidate_reference
#             break

#     if statement_no is None:
#         statement_match = re.search(r"Statement\s+([A-Z0-9.\-]+)", text, re.IGNORECASE)
#         if statement_match:
#             statement_no = statement_match.group(1)

#     if reference_no is None:
#         reference_match = re.search(
#             r"Statement\s+[A-Z0-9.\-]+\s*([A-Z][A-Z0-9.\-]+?)(?=\s*\d{2}/\d{2}/\d{4})",
#             text,
#         )
#         if reference_match:
#             reference_no = reference_match.group(1)

#     date_match = re.search(r"(\d{2}/\d{2}/\d{4})", text)
#     time_match = re.search(r"\d{2}:\d{2}", text)
#     date = date_match.group(1) if date_match else ""
#     time = time_match.group(0) if time_match else ""
#     table_match = re.search(r"(?:Tafel|Table)\s*(\d+)", text, re.IGNORECASE)
#     table = table_match.group(1) if table_match else None
#     total_match = re.search(r"€\s*([\d.,]+)", text)
#     total_amount = float(total_match.group(1).replace(",", ".")) if total_match else None
#     tax_match = re.search(r"Btw[: ]+([A-Z0-9]+)", text, re.IGNORECASE)
#     tax_id = tax_match.group(1) if tax_match else None

#     non_item_keywords = {
#         "amount",
#         "draft",
#         "receipt",
#         "floor plan",
#         "division",
#         "payment",
#         "btw",
#         "tax",
#         "dank",
#         "lightspeed",
#         "amstelveen",
#         "statement",
#         "reference",
#         "device",
#         "employee",
#         "table",
#         "taf",
#         "thank",
#         "visit",
#         "no payment",
#         "hint",
#         "nl",
#     }

#     items: List[Dict[str, Optional[float]]] = []
#     for row in rows:
#         center_y = row["center_y"]
#         if table_baseline is not None and center_y <= table_baseline:
#             continue
#         if center_y >= max_price_center_y + 18:
#             continue

#         row_tokens = row["tokens"]
#         cutoff = price_column_x - 4
#         left_words = [t.text for t in row_tokens if t.x + t.width <= cutoff]
#         if not left_words:
#             continue

#         possible_price = None
#         for candidate in reversed(row_tokens):
#             if candidate.x >= price_column_x - 6 and re.fullmatch(r"\d+[.,]\d{2}", candidate.text):
#                 possible_price = float(candidate.text.replace(",", "."))
#                 break
#         if possible_price is None:
#             continue

#         name = " ".join(left_words).strip(" :-")
#         name_lower = name.lower()
#         if len(name) < 2:
#             continue
#         if not re.search(r"[a-z]", name_lower):
#             continue
#         if any(keyword in name_lower for keyword in non_item_keywords):
#             continue

#         items.append({"name": name, "price": round(possible_price, 2)})

#     if total_amount is None and items:
#         prices = [item["price"] for item in items if item["price"] is not None]
#         total_amount = round(sum(prices), 2) if prices else None

#     receipt_data = {
#         "store_name": store_name,
#         "statement_no": statement_no,
#         "reference_no": reference_no,
#         "order_device": order_device,
#         "employee_no": employee_no,
#         "date": date,
#         "time": time,
#         "table": table,
#         "items": items,
#         "total_amount": total_amount,
#         "tax_id": tax_id,
#     }

#     markdown_lines = [f"# {store_name}", ""]
#     if date or time:
#         markdown_lines.append(f"**Date:** {date} {time}".strip())
#     if table:
#         markdown_lines.append(f"**Table:** {table}")
#     if statement_no:
#         markdown_lines.append(f"**Statement:** {statement_no}")
#     if reference_no:
#         markdown_lines.append(f"**Reference:** {reference_no}")
#     assignment_parts = []
#     if order_device:
#         assignment_parts.append(f"Device: {order_device}")
#     if employee_no:
#         assignment_parts.append(f"Employee: {employee_no}")
#     if assignment_parts:
#         markdown_lines.append("**Assignment:** " + " | ".join(assignment_parts))
#     markdown_lines.append("\n---\n")
#     markdown_lines.append("| Item | Price (€) |\n|------|-----------:|")
#     for item in items:
#         price_text = f"{item['price']:.2f}" if item["price"] is not None else ""
#         markdown_lines.append(f"| {item['name']} | {price_text} |")
#     if total_amount is not None:
#         markdown_lines.append(f"| **Total** | **{total_amount:.2f}** |")
#     markdown_lines.append("")
#     if tax_id:
#         markdown_lines.append(f"**Tax ID:** {tax_id}")
#     markdown_text = "\n".join(markdown_lines)

#     return receipt_data, markdown_text

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import re
import statistics
import math
from google.cloud import vision

@dataclass
class OCRToken:
    text: str
    x: float
    y: float
    width: float
    height: float

def parse_receipt_text(full_text: str, response=None) -> Tuple[Dict, str]:
    """
    Convert OCR text into structured data (schema) + markdown view.
    Uses layout-based boundaries to extract items reliably:
    - Items are ALWAYS between the table start line and amount boundary
    - Uses bounding boxes for stable parsing regardless of item content changes
    """
    if response is None or not response.full_text_annotation.pages:
        raise ValueError("Vision response with layout required for stable parsing.")

    page = response.full_text_annotation.pages[0]
    print(f"full_text_annotation: {full_text_annotation}")

    def token_stream():
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = "".join(symbol.text for symbol in word.symbols)
                    vertices = word.bounding_box.vertices
                    xs = [v.x for v in vertices]
                    ys = [v.y for v in vertices]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    yield OCRToken(
                        text=word_text,
                        x=x_min,
                        y=y_min,
                        width=x_max - x_min,
                        height=y_max - y_min,
                    )

    tokens: List[OCRToken] = list(token_stream())
    print(f"[parse_receipt_text] total_tokens={len(tokens)}")
    tokens.sort(key=lambda t: (round(t.y / 6), t.x))

    # ========== FIND STRUCTURAL BOUNDARIES ==========
    
    # Find the "Tafel"/"Table" row (marks START of item table)
    table_start_y: Optional[float] = None
    for idx, token in enumerate(tokens):
        if token.text.lower() in ["tafel", "table"]:
            table_start_y = token.y
            # Get the center Y of this row for better boundary
            row_tokens = [t for t in tokens if abs(t.y - token.y) <= 10]
            table_start_y = statistics.mean(t.y + t.height / 2 for t in row_tokens)
            print(f"[parse_receipt_text] table_start_y={table_start_y:.1f}")
            break
    
    # Find the "Amount due" row (marks END of item table)
    amount_boundary: Optional[float] = None
    for idx, token in enumerate(tokens):
        if "amount" in token.text.lower() and "due" in tokens[idx+1].text.lower() if idx+1 < len(tokens) else False:
            row_tokens = [t for t in tokens if abs(t.y - token.y) <= 10]
            amount_boundary = min(t.y for t in row_tokens)  # Top of "Amount due" row
            print(f"[parse_receipt_text] amount_boundary={amount_boundary:.1f}")
            break
    
    if amount_boundary is None:
        # Fallback: look for any "amount" keyword
        for token in tokens:
            if "amount" in token.text.lower():
                amount_boundary = token.y
                print(f"[parse_receipt_text] amount_boundary (fallback)={amount_boundary:.1f}")
                break
    
    if table_start_y is None:
        table_start_y = 0.0
    if amount_boundary is None:
        amount_boundary = float("inf")
    
    print(f"[parse_receipt_text] item_zone: y=[{table_start_y:.1f}, {amount_boundary:.1f}]")

    # ========== CLUSTER TOKENS INTO ROWS ==========
    
    row_clusters: List[List[OCRToken]] = []
    for token in tokens:
        placed = False
        for cluster in row_clusters:
            anchor = cluster[0]
            anchor_center = anchor.y + anchor.height / 2
            token_center = token.y + token.height / 2
            if abs(anchor_center - token_center) <= max(anchor.height, token.height) * 0.45:
                cluster.append(token)
                placed = True
                break
        if not placed:
            row_clusters.append([token])

    rows: List[Dict[str, Any]] = []
    for cluster in row_clusters:
        cluster.sort(key=lambda t: t.x)
        center_y = statistics.mean(t.y + t.height / 2 for t in cluster)
        rows.append({"tokens": cluster, "center_y": center_y})
    
    print(f"[parse_receipt_text] total_row_count={len(rows)}")

    # ========== IDENTIFY PRICE COLUMNS ==========
    
    price_regex = re.compile(r"\d+[.,]\d{2}")
    
    # Find price tokens ONLY in the item zone
    price_tokens_in_zone = [
        t for t in tokens
        if price_regex.fullmatch(t.text)
        and (table_start_y <= (t.y + t.height / 2) < amount_boundary)
    ]
    
    if not price_tokens_in_zone:
        raise ValueError(f"No price tokens found in item zone [{table_start_y:.1f}, {amount_boundary:.1f}]")
    
    print(f"[parse_receipt_text] price_tokens_in_zone={len(price_tokens_in_zone)}")
    
    # Cluster price tokens by X position (vertical columns)
    column_tolerance = 14.0
    column_clusters: List[Dict[str, float]] = []
    for tok in price_tokens_in_zone:
        center = tok.x + tok.width / 2
        assigned = False
        for cluster in column_clusters:
            if abs(cluster["center"] - center) <= column_tolerance:
                cluster["sum"] += center
                cluster["count"] += 1
                cluster["center"] = cluster["sum"] / cluster["count"]
                assigned = True
                break
        if not assigned:
            column_clusters.append({"center": center, "sum": center, "count": 1})
    
    price_columns = sorted(cluster["center"] for cluster in column_clusters)
    print(f"[parse_receipt_text] price_columns={price_columns}")
    
    unit_col_index = 0
    total_col_index = len(price_columns) - 1

    # ========== EXTRACT HEADER METADATA ==========
    
    text = full_text.replace("\r", "")
    store_match = re.search(r"(Fluffy\s+Cafe-?Restaur[a-z]*)", text, re.IGNORECASE)
    store_name = (
        re.sub(r"\s+", " ", store_match.group(1)).strip()
        if store_match
        else "Fluffy Cafe-Restaurant"
    )
    
    terminal_match = re.search(r"([A-Za-z][A-Za-z0-9]*)\s*/\s*([A-Za-z0-9\-]+)", text)
    order_device = terminal_match.group(1) if terminal_match else None
    employee_no = terminal_match.group(2) if terminal_match else None

    statement_no: Optional[str] = None
    reference_no: Optional[str] = None
    for idx, token in enumerate(tokens):
        if token.text.lower() == "statement":
            if idx + 1 < len(tokens):
                candidate_statement = tokens[idx + 1].text.strip(":")
                if re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]*", candidate_statement):
                    statement_no = candidate_statement
            if idx + 2 < len(tokens):
                candidate_reference = tokens[idx + 2].text.strip(":")
                if (
                    re.fullmatch(r"[A-Z][A-Z0-9.\-]*", candidate_reference)
                    and "/" not in candidate_reference
                ):
                    reference_no = candidate_reference
            break

    if statement_no is None:
        statement_match = re.search(r"Statement\s+([A-Z0-9.\-]+)", text, re.IGNORECASE)
        if statement_match:
            statement_no = statement_match.group(1)

    if reference_no is None:
        reference_match = re.search(
            r"Statement\s+[A-Z0-9.\-]+\s*([A-Z][A-Z0-9.\-]+?)(?=\s*\d{2}/\d{2}/\d{4})",
            text,
        )
        if reference_match:
            reference_no = reference_match.group(1)

    date_match = re.search(r"(\d{2}/\d{2}/\d{4})", text)
    time_match = re.search(r"\d{2}:\d{2}", text)
    date = date_match.group(1) if date_match else ""
    time = time_match.group(0) if time_match else ""
    
    table_match = re.search(r"(?:Tafel|Table)\s+(\d+)", text, re.IGNORECASE)
    table = table_match.group(1) if table_match else None
    
    total_match = re.search(r"€\s*([\d.,]+)", text)
    total_amount = float(total_match.group(1).replace(",", ".")) if total_match else None
    
    tax_match = re.search(r"Btw[: ]+([A-Z0-9]+)", text, re.IGNORECASE)
    tax_id = tax_match.group(1) if tax_match else None

    # ========== EXTRACT ITEMS FROM BOUNDED ZONE ==========
    
    non_item_keywords = {
        "amount",
        "draft",
        "receipt",
        "floor plan",
        "division",
        "payment",
        "btw",
        "tax",
        "dank",
        "lightspeed",
        "amstelveen",
        "statement",
        "reference",
        "device",
        "employee",
        "table",
        "tafel",
        "thank",
        "visit",
        "no payment",
        "hint",
        "nl",
    }

    skip_next_row = False
    items: List[Dict[str, Any]] = []
    
    for idx, row in enumerate(rows):
        center_y = row["center_y"]
        row_text = " ".join(tok.text for tok in row["tokens"])
        
        # ===== BOUNDARY CHECKS: Only process rows in item zone =====
        if center_y <= table_start_y:
            print(f"[parse_receipt_text] skip_before_table row_index={idx} y={center_y:.1f}")
            continue
        if center_y >= amount_boundary:
            print(f"[parse_receipt_text] stop_at_amount row_index={idx} y={center_y:.1f}")
            break  # Stop processing further rows
        
        print(f"[parse_receipt_text] row_index={idx} y={center_y:.1f} text='{row_text}'")
        
        if skip_next_row:
            print(f"[parse_receipt_text] skip_due_to_hint row_index={idx}")
            skip_next_row = False
            continue

        row_text_lower = row_text.lower()
        if "hint" in row_text_lower:
            print(f"[parse_receipt_text] found_hint row_index={idx}, skipping next row")
            skip_next_row = True
            continue

        # ===== EXTRACT PRICES FROM THIS ROW =====
        row_tokens = row["tokens"]
        row_price_tokens: List[Tuple[OCRToken, float, Optional[int]]] = []
        
        for tok in row_tokens:
            if price_regex.fullmatch(tok.text):
                center_x = tok.x + tok.width / 2
                nearest_idx = min(
                    range(len(price_columns)),
                    key=lambda c_idx: abs(price_columns[c_idx] - center_x),
                )
                distance = abs(price_columns[nearest_idx] - center_x)
                col_idx = nearest_idx if distance <= column_tolerance else None
                value = float(tok.text.replace(",", "."))
                row_price_tokens.append((tok, value, col_idx))
        
        if not row_price_tokens:
            print(f"[parse_receipt_text] row_index={idx} no_prices, skipping")
            continue

        # ===== EXTRACT ITEM NAME AND QUANTITY =====
        left_tokens = [
            tok for tok in row_tokens
            if not price_regex.fullmatch(tok.text) and tok.text != "€"
        ]
        
        if not left_tokens:
            print(f"[parse_receipt_text] row_index={idx} no_left_tokens")
            continue

        name_tokens: List[str] = []
        quantity: Optional[int] = None
        
        for tok in left_tokens:
            cleaned = tok.text.strip()
            if not cleaned:
                continue
            if quantity is None:
                qty_match = re.fullmatch(r"(\d+)(?:x)?", cleaned)
                if qty_match:
                    quantity = int(qty_match.group(1))
                    print(f"[parse_receipt_text] row_index={idx} quantity={quantity}")
                    continue
            name_tokens.append(cleaned)

        if not name_tokens:
            print(f"[parse_receipt_text] row_index={idx} no_name_tokens")
            continue

        name = " ".join(name_tokens).strip(" :-")
        name_lower = name.lower()
        
        if len(name) < 2 or not re.search(r"[a-z]", name_lower):
            print(f"[parse_receipt_text] row_index={idx} invalid_name='{name}'")
            continue
        
        if any(keyword in name_lower for keyword in non_item_keywords):
            print(f"[parse_receipt_text] row_index={idx} filtered_by_keyword='{name}'")
            continue

        # ===== ASSIGN PRICES TO COLUMNS =====
        totals_in_row = [
            value for _, value, col_idx in row_price_tokens if col_idx == total_col_index
        ]
        units_in_row = [
            value for _, value, col_idx in row_price_tokens if col_idx == unit_col_index
        ]
        unclassified = [
            value for _, value, col_idx in row_price_tokens if col_idx is None
        ]
        
        print(f"[parse_receipt_text] row_index={idx} units={units_in_row} totals={totals_in_row}")

        unit_price: Optional[float] = None
        total_price: Optional[float] = None

        if len(price_columns) == 1:
            total_price = row_price_tokens[-1][1]
            if quantity and quantity > 0:
                unit_price = round(total_price / quantity, 2)
        else:
            if totals_in_row:
                total_price = totals_in_row[-1]
            elif unclassified:
                total_price = unclassified[-1]
            else:
                total_price = row_price_tokens[-1][1]
            
            if units_in_row:
                unit_price = units_in_row[0]
            elif quantity and quantity > 0:
                unit_price = round(total_price / quantity, 2)

        if total_price is None:
            print(f"[parse_receipt_text] row_index={idx} no_total_price")
            continue

        # ===== CREATE ITEM ENTRY =====
        item_entry: Dict[str, Any] = {"name": name, "price": round(total_price, 2)}
        if quantity is not None:
            item_entry["quantity"] = quantity
        if unit_price is not None:
            unit_price_rounded = round(unit_price, 2)
            if not math.isclose(unit_price_rounded, item_entry["price"], rel_tol=1e-9):
                item_entry["unit_price"] = unit_price_rounded

        print(f"[parse_receipt_text] row_index={idx} ADDED item={item_entry}")
        items.append(item_entry)

    print(f"[parse_receipt_text] final_item_count={len(items)}")

    # ===== FALLBACK: Sum items if total not found =====
    if total_amount is None and items:
        totals = [item["price"] for item in items if item.get("price") is not None]
        if totals:
            total_amount = round(sum(totals), 2)

    # ========== BUILD STRUCTURED DATA ==========
    
    receipt_data = {
        "store_name": store_name,
        "statement_no": statement_no,
        "reference_no": reference_no,
        "order_device": order_device,
        "employee_no": employee_no,
        "date": date,
        "time": time,
        "table": table,
        "items": items,
        "total_amount": total_amount,
        "tax_id": tax_id,
    }

    # ========== BUILD MARKDOWN VIEW ==========
    
    markdown_lines = [f"# {store_name}", ""]
    
    if date or time:
        markdown_lines.append(f"**Date:** {date} {time}".strip())
    if table:
        markdown_lines.append(f"**Table:** {table}")
    if statement_no:
        markdown_lines.append(f"**Statement:** {statement_no}")
    if reference_no:
        markdown_lines.append(f"**Reference:** {reference_no}")
    
    assignment_parts = []
    if order_device:
        assignment_parts.append(f"Device: {order_device}")
    if employee_no:
        assignment_parts.append(f"Employee: {employee_no}")
    if assignment_parts:
        markdown_lines.append("**Assignment:** " + " | ".join(assignment_parts))
    
    markdown_lines.append("\n---\n")

    has_quantities = any("quantity" in item for item in items)
    has_unit_prices = any("unit_price" in item for item in items)

    header_cells = ["Item"]
    align_cells = [":---"]
    if has_quantities:
        header_cells.insert(0, "Qty")
        align_cells.insert(0, "---:")
    if has_unit_prices:
        header_cells.append("Unit (€)")
        align_cells.append("---:")
    header_cells.append("Total (€)")
    align_cells.append("---:")

    markdown_lines.append("| " + " | ".join(header_cells) + " |")
    markdown_lines.append("|" + "|".join(align_cells) + "|")

    for item in items:
        row_cells: List[str] = []
        if has_quantities:
            row_cells.append("" if "quantity" not in item else str(item["quantity"]))
        row_cells.append(item["name"])
        if has_unit_prices:
            row_cells.append("" if "unit_price" not in item else f"{item['unit_price']:.2f}")
        row_cells.append("" if item.get("price") is None else f"{item['price']:.2f}")
        markdown_lines.append("| " + " | ".join(row_cells) + " |")

    if total_amount is not None:
        footer_cells: List[str] = []
        if has_quantities:
            footer_cells.append("")
        footer_cells.append("**Total**")
        if has_unit_prices:
            footer_cells.append("")
        footer_cells.append(f"**{total_amount:.2f}**")
        markdown_lines.append("| " + " | ".join(footer_cells) + " |")

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
