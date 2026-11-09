import os
from dotenv import load_dotenv

load_dotenv()

import mysql.connector
import torch
import torch.nn as nn
from fastapi import FastAPI, APIRouter, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from typing import Optional
from datetime import date, timedelta

import shutil
from fastapi import UploadFile, File, HTTPException
import easyocr

# ==========================================
# 1. INISIALISASI UTAMA FASTAPI
# ==========================================
app = FastAPI(
    title="Zaini.dev Portfolio & AI-WMS Engine",
    version="2.0",
    description="Unified Backend API server for portfolio console, MySQL databases, PyTorch models, and AI-WMS chatbot assistant."
)

# Konfigurasi CORS (Wajib agar dapat diakses oleh Laravel di port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 2. KONFIGURASI DATABASE PORTFOLIO (MySQL Port 3308)
# ==========================================
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

SQLALCHEMY_DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserModel(Base):
    __tablename__ = "users_biodata"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100))
    role = Column(String(100))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==========================================
# 3. KONEKSI DATABASE AI-WMS (zaini_ai_wms di Port 3308)
# ==========================================
def get_ai_wms_db():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )
print("DB_HOST =", os.getenv("DB_HOST"))
print("DB_NAME =", os.getenv("DB_NAME"))


# ==========================================
# 4. PYDANTIC SCHEMAS
# ==========================================
class PredictionInput(BaseModel):
    feature_1: float
    feature_2: float

class PromptInput(BaseModel):
    prompt: str
    max_tokens: int = 50

class LLMQueryRequest(BaseModel):
    warehouse_id: Optional[int] = None
    prompt_query: str


# ==========================================
# 5. ENDPOINT UTAMA PORTFOLIO & AI GENERAL
# ==========================================

@app.get("/")
def root_index():
    return {
        "status": "online",
        "author": "Zaini",
        "system": "Zaini.dev Full Portfolio & AI-WMS Engine Active",
        "endpoints_available": [
            "/api/v1/health",
            "/api/v1/biodata",
            "/api/v1/predict",
            "/api/v1/generate-ai",
            "/api/v1/ai-wms/llm-assistant",
            "/api/v1/ai-wms/forecast/{warehouse_id}/{sku_id}"
        ]
    }

@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "server": "Python Uvicorn", "code": 200}

@app.get("/api/v1/biodata")
def get_biodata(
    id: Optional[int] = Query(None, description="Cari user berdasarkan ID"),
    db: Session = Depends(get_db)
):
    try:
        if id is not None:
            user = db.query(UserModel).filter(UserModel.id == id).first()
        else:
            user = db.query(UserModel).first()
        
        if not user:
            return {
                "status": "success",
                "source": "database_mysql",
                "message": f"Data dengan ID {id} tidak ditemukan di database.",
                "data": None
            }
            
        return {
            "status": "success",
            "source": "database_mysql",
            "query_received": {"id": id},
            "data": user
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "data": None
        }

@app.post("/api/v1/generate-ai")
def generate_ai_response(payload: PromptInput):
    try:
        return {
            "status": "success",
            "engine": "Zaini.dev LLM Simulator v1.0",
            "prompt_received": payload.prompt,
            "simulated_response": f"Analisis Prompt berhasil diproses. Output: '[Echo] Berdasarkan instruksi Anda mengenai '{payload.prompt}', sistem menjalankan inferensi teks dengan optimal.'",
            "tokens_used": len(payload.prompt.split()) + 15
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 6. ENDPOINT MODUL AI-WMS & CHATBOT BOOTH
# ==========================================

@app.get("/api/v1/ai-wms/warehouses")
def get_all_warehouses():
    conn = get_ai_wms_db()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM warehouses")
        result = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()
        
    return {
        "status": "success",
        "total_data": len(result),
        "data": result
    }

@app.post("/api/v1/ai-wms/llm-assistant")
def generative_wms_assistant(payload: LLMQueryRequest):
    try:
        conn = get_ai_wms_db()
        cursor = conn.cursor(dictionary=True)
        
        # 1. Ambil daftar warehouse
        try:
            cursor.execute("SELECT * FROM warehouse")
        except Exception:
            cursor.execute("SELECT * FROM warehouses")
        warehouses = cursor.fetchall()
        
        # 2. Ambil master data SKUs (SELECT * FROM skus)
        skus_data = []
        try:
            cursor.execute("SELECT * FROM skus")
            skus_data = cursor.fetchall()
        except Exception:
            pass
        
        # 3. Ambil data stok gabungan
        all_stocks = []
        try:
            cursor.execute("""
                SELECT w.warehouse_name, s.sku_code, s.product_name, ist.quantity, ist.min_stock_level 
                FROM inventory_stocks ist
                JOIN warehouse w ON ist.warehouse_id = w.id
                JOIN skus s ON ist.sku_id = s.id
            """)
            all_stocks = cursor.fetchall()
        except Exception:
            try:
                cursor.execute("""
                    SELECT w.warehouse_name, s.sku_code, s.product_name, ist.quantity, ist.min_stock_level 
                    FROM inventory_stocks ist
                    JOIN warehouses w ON ist.warehouse_id = w.id
                    JOIN skus s ON ist.sku_id = s.id
                """)
                all_stocks = cursor.fetchall()
            except Exception:
                pass

        critical_stocks = [item for item in all_stocks if item.get('quantity', 0) <= item.get('min_stock_level', 0)]
        total_units = sum([item.get('quantity', 0) for item in all_stocks])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()
    
    # Format nama warehouse
    wh_names_list = []
    for w in warehouses:
        name = w.get("warehouse_name") or w.get("name") or f"Warehouse {w.get('id')}"
        loc = w.get("location", "")
        wh_names_list.append(f"{name} {f'({loc})' if loc else ''}".strip())

    context_data = {
        "total_warehouses": len(warehouses),
        "warehouse_list": wh_names_list,
        "total_sku_types": len(skus_data) if skus_data else len(all_stocks),
        "total_stock_units": total_units,
        "critical_items_count": len(critical_stocks)
    }
    
    # --- PENGECEKAN INTENT YANG LEBIH SPESIFIK & URUTAN DIDAHULUKAN ---
    query_lower = payload.prompt_query.lower()
    
    # 1. Intent Rincian / Nama Barang (Ditaruh paling atas agar terdeteksi duluan)
    # 1. Intent Rincian / Nama Barang (Format Bernomor & Rapi ke Bawah)
    if any(k in query_lower for k in ["nama barang", "sebutkan barang", "rincian barang", "daftar barang", "list barang", "sku", "produk", "jenis barang"]):
        if skus_data:
            list_items = []
            for idx, s in enumerate(skus_data, 1):
                name = s.get('product_name') or s.get('name')
                code = s.get('sku_code') or s.get('code')
                list_items.append(f"{idx}. {name} (Kode: {code})")
            
            sku_details = "<br>".join(list_items)
            insight_text = f"Berikut adalah daftar dan rincian nama barang/SKU yang tersedia dalam sistem (Total {len(skus_data)} jenis SKU):<br><br>{sku_details}"
        elif all_stocks:
            list_items = []
            for idx, i in enumerate(all_stocks, 1):
                list_items.append(f"{idx}. {i.get('product_name')} di {i.get('warehouse_name')} (Stok: {i.get('quantity')})")
            item_details = "<br>".join(list_items)
            insight_text = f"Berikut adalah daftar barang di inventaris:<br><br>{item_details}"
        else:
            insight_text = "Belum ada data barang atau SKU yang terdaftar di dalam database."
            
    # 2. Intent Warehouse / Gudang
    elif any(k in query_lower for k in ["warehouse", "gudang", "tempat", "cabang"]):
        if any(k in query_lower for k in ["berapa", "jumlah"]):
            insight_text = f"Total keseluruhan terdapat {len(warehouses)} warehouse aktif dalam sistem, yaitu: {', '.join(wh_names_list)}."
        else:
            insight_text = f"Berikut adalah daftar warehouse aktif dalam sistem jaringan kami: {', '.join(wh_names_list)}."
            
    # 3. Intent Stok Kurang Dari / Batas Tertentu
    elif any(k in query_lower for k in ["kurang dari", "stok kurang", "bawah", "<", "kecil"]):
        import re
        numbers = re.findall(r'\d+', query_lower)
        limit_val = int(numbers[0]) if numbers else 5
        
        filtered_items = [item for item in all_stocks if item.get('quantity', 0) < limit_val]
        if filtered_items:
            items_desc = ", ".join([f"{i['product_name']} di {i['warehouse_name']} (Sisa: {i['quantity']} unit)" for i in filtered_items])
            insight_text = f"Berikut adalah barang yang stoknya kurang dari {limit_val}: {items_desc}."
        else:
            insight_text = f"Tidak ada barang yang stoknya di bawah {limit_val} unit."
            
    # 4. Intent Stok Kritis
    elif any(k in query_lower for k in ["kritis", "restock", "habis", "minimum"]):
        if critical_stocks:
            item_names = ", ".join([f"{item.get('product_name')} di {item.get('warehouse_name')} (Sisa: {item.get('quantity')})" for item in critical_stocks])
            insight_text = f"PERINGATAN STOK KRITIS: Item berikut berada di bawah batas minimum: {item_names}."
        else:
            insight_text = "Status seluruh inventaris saat ini terpantau aman dan berada di atas batas minimum."
            
    # 5. Default Ringkasan Umum
    else:
        insight_text = (
            f"Analisis AI-WMS: Sistem mengelola {len(warehouses)} warehouse aktif dengan total {len(skus_data) if skus_data else len(all_stocks)} jenis SKU "
            f"dan {total_units} unit stok fisik. Silakan tanyakan rincian barang, daftar SKU, atau status stok."
        )

    return {
        "status": "success",
        "engine": "FastAPI + MySQL + PyTorch Tensor / GenAI",
        "user_query": payload.prompt_query,
        "database_context": context_data,
        "critical_alerts": critical_stocks,
        "llm_insight": insight_text
    }

@app.get("/api/v1/ai-wms/forecast/{warehouse_id}/{sku_id}")
def predict_demand_tensor(warehouse_id: int, sku_id: int):
    conn = get_ai_wms_db()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT ist.quantity, ist.min_stock_level, s.product_name 
            FROM inventory_stocks ist
            JOIN skus s ON ist.sku_id = s.id
            WHERE ist.warehouse_id = %s AND ist.sku_id = %s
        """, (warehouse_id, sku_id))
        row = cursor.fetchone()
    finally:
        cursor.close()
        conn.close()
        
    if not row:
        raise HTTPException(status_code=404, detail="Data stok atau SKU tidak ditemukan di gudang ini.")
    
    input_tensor = torch.tensor([[float(row["quantity"]), float(row["min_stock_level"])]], dtype=torch.float32)
    weights = torch.tensor([[1.2], [0.8]], dtype=torch.float32)
    
    with torch.no_grad():
        tensor_output = torch.matmul(input_tensor, weights)
        predicted_demand = round(tensor_output.item(), 2)

    return {
        "status": "success",
        "model_type": "PyTorch Tensor Calculation",
        "product_name": row["product_name"],
        "tensor_shape": list(input_tensor.shape),
        "current_quantity": row["quantity"],
        "predicted_next_period_demand": predicted_demand
    }

# ==========================================
# 6. ENDPOINT MODUL SCAN OCR
# ==========================================
# Inisialisasi reader EasyOCR (mendukung bahasa Inggris & Indonesia)


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/v1/ai-wms/ocr-scan")
async def scan_ocr_image(file: UploadFile = File(...)):

    reader = easyocr.Reader(['en', 'id'])
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Proses OCR menggunakan EasyOCR
        results = reader.readtext(file_path, detail=0)
        extracted_text = "\n".join(results)

        if os.path.exists(file_path):
            os.remove(file_path)

        return {
            "status": "success",
            "filename": file.filename,
            "extracted_text": extracted_text.strip() if extracted_text else "Tidak ada teks yang terdeteksi pada gambar."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR Processing Error: {str(e)}")