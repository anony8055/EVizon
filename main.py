from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import os, uuid, zipfile, re
from supabase_client import supabase
from gradio_client import Client, handle_file

app = FastAPI()

UPLOAD_DIR = "uploads"
ALBUM_DIR = "albums"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ALBUM_DIR, exist_ok=True)

# --- CONFIGURATION ---
# REPLACE THIS URL with your actual Hugging Face Space URL!
# Example: "username/evizon-ai-worker"
HF_SPACE_ID = "anon8055/evizon-ai-worker" 
hf_client = Client(HF_SPACE_ID)

# ---------- Helper functions ----------
def extract_year_from_text(text: str) -> str:
    if not text: return "Unknown"
    match = re.search(r"(20\d{2})", text)
    return match.group(1) if match else "Unknown"

# ---------- API Endpoints ----------

@app.get("/")
def home():
    return {"message": "Evizon backend active (Lightweight Mode)."}

@app.post("/upload")
async def upload_photo(file: UploadFile = File(...)):
    # 1. Save locally first
    img_bytes = await file.read()
    photo_id = str(uuid.uuid4())
    filename = file.filename
    temp_path = f"{UPLOAD_DIR}/{photo_id}.jpg"

    with open(temp_path, "wb") as f:
        f.write(img_bytes)

    # 2. Send to Hugging Face for Analysis
    print(f"Sending {filename} to AI Worker...")
    try:
        # This calls the 'predict' function on your HF Space
        result = hf_client.predict(
            image_path=handle_file(temp_path),
            api_name="/predict"
        )
        # Result is already a JSON dictionary (tags)
        tags = result
    except Exception as e:
        print(f"AI Error: {e}")
        return {"error": "AI Processing Failed", "details": str(e)}

    # 3. Process results
    event_name = tags.get("event_text", "Unknown")
    year = extract_year_from_text(event_name)

    # 4. Organize files
    event_folder = f"{ALBUM_DIR}/{event_name}/{year}"
    os.makedirs(event_folder, exist_ok=True)
    final_path = f"{event_folder}/{filename}"
    
    # Move file (rewrite bytes)
    with open(final_path, "wb") as f:
        f.write(img_bytes)

    # 5. Insert to Supabase
    supabase.table("event_photos").insert({
        "id": photo_id,
        "photo_url": final_path,
        "event_name": event_name,
        "department": "Unknown",
        "event_date": None,
        "category": "general",
        "cluster_id": -1, # Clustering disabled to save RAM
        "is_blurry": tags.get("is_blurry", False),
        "hash": tags.get("hash", "")
    }).execute()

    os.remove(temp_path)
    return {"status": "uploaded", "event": event_name, "year": year}

# --- LIGHTWEIGHT SEARCH ENDPOINTS ---
# (These don't use RAM, so they are fine to keep)

@app.get("/filter")
def filter_photos(event: str = None, department: str = None, date: str = None):
    query = supabase.table("event_photos").select("*")
    if event: query = query.ilike("event_name", f"%{event}%")
    if department: query = query.eq("department", department)
    if date: query = query.eq("event_date", date)
    return {"results": query.execute().data}

@app.get("/search")
def natural_search(q: str = Query(...)):
    # Simplified search logic
    q_lower = q.lower()
    dept_match = re.search(r"\b(cse|ece|eee|me)\b", q_lower)
    department = dept_match.group(1).upper() if dept_match else None
    
    query = supabase.table("event_photos").select("*")
    if department: query = query.eq("department", department)
    
    # Basic text match on event name since we don't have the clean function locally anymore
    # (Or you can duplicate the simple clean string function here if you want)
    if not department:
         query = query.ilike("event_name", f"%{q}%")
         
    results = query.execute().data
    return {"query": q, "results": results}

@app.get("/uploads/{filename}")
def serve_image(filename: str):
    # This might break if files are moved to 'albums/...' 
    # but keeping it for compatibility with your frontend
    path = os.path.join(UPLOAD_DIR, filename) 
    if not os.path.exists(path): return JSONResponse({"error": "Not found"}, 404)
    return FileResponse(path)

# NOTE: /cluster endpoint removed because it requires Pytorch/GPU
# If you need it for the demo, use the 'ngrok' method instead.