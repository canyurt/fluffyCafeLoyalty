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
    # print(f"full_text_annotation: {full_text_annotation}")
    annotation = response.full_text_annotation
    print(f"[parse_receipt_text] full_text_annotation present={annotation is not None}")    

    # def token_stream():
    #     for block in page.blocks:
    #         for paragraph in block.paragraphs:
    #             for word in paragraph.words:
    #                 word_text = "".join(symbol.text for symbol in word.symbols)
    #                 vertices = word.bounding_box.vertices
    #                 xs = [v.x for v in vertices]
    #                 ys = [v.y for v in vertices]
    #                 x_min, x_max = min(xs), max(xs)
    #                 y_min, y_max = min(ys), max(ys)
    #                 yield OCRToken(
    #                     text=word_text,
    #                     x=x_min,
    #                     y=y_min,
    #                     width=x_max - x_min,
    #                     height=y_max - y_min,
    #                 )

    # tokens: List[OCRToken] = list(token_stream())

    # # NEW: normalize coordinates if Vision reported them in a 90° clockwise frame
    # # tall_tokens = sum(1 for t in tokens if t.height > t.width * 1.4)
    # # rotated_clockwise = tall_tokens / max(len(tokens), 1) > 0.6

    # # if rotated_clockwise:
    # #     page_width = page.width
    # #     for t in tokens:
    # #         original_x, original_y = t.x, t.y
    # #         t.x = original_y
    # #         t.y = page_width - (original_x + t.width)
    # #         t.width, t.height = t.height, t.width
    
    # # Detect Vision’s 90° clockwise reporting and swap axes if needed
    # tall_tokens = sum(1 for t in tokens if t.height > t.width * 1.4)
    # rotated_clockwise = tall_tokens / max(len(tokens), 1) > 0.6

    # if rotated_clockwise:
    #     for t in tokens:
    #         original_x, original_y = t.x, t.y
    #         t.x = original_y            # use the original column value
    #         t.y = original_x            # use the original row value
    #         t.width, t.height = t.height, t.width

    raw_tokens: List[Tuple[str, List[Tuple[int, int]]]] = []

    for block in page.blocks:
        for paragraph in block.paragraphs:
            for word in paragraph.words:
                word_text = "".join(symbol.text for symbol in word.symbols)
                vertices = [(v.x, v.y) for v in word.bounding_box.vertices]
                raw_tokens.append((word_text, vertices))

    # Detect the 90° clockwise normalisation Vision applies to tilted pages
    tall_tokens = 0
    for _, vertices in raw_tokens:
        xs = [vx for vx, _ in vertices]
        ys = [vy for _, vy in vertices]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        if height > width * 1.4:
            tall_tokens += 1

    rotated_clockwise = tall_tokens / max(len(raw_tokens), 1) > 0.6

    tokens: List[OCRToken] = []
    for word_text, vertices in raw_tokens:
        if rotated_clockwise:
            transformed = [(page.width - vy, vx) for vx, vy in vertices]  # rotate back
        else:
            transformed = vertices
        xs = [vx for vx, _ in transformed]
        ys = [vy for _, vy in transformed]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        tokens.append(
            OCRToken(
                text=word_text,
                x=float(x_min),
                y=float(y_min),
                width=float(x_max - x_min),
                height=float(y_max - y_min),
            )
        )



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
            # if abs(anchor_center - token_center) <= max(anchor.height, token.height) * 0.45:
            if abs(anchor_center - token_center) <= max(anchor.height, token.height) * 0.60:
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
