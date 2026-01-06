from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from gradio_client import Client, handle_file
from supabase_client import supabase  # Make sure this file exists and is correct
import os, uuid, zipfile, re, shutil

app = FastAPI()

UPLOAD_DIR = "uploads"
ALBUM_DIR = "albums"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ALBUM_DIR, exist_ok=True)

# --- CONFIGURATION ---
# REPLACE THIS WITH YOUR EXACT HUGGING FACE SPACE ID
# Example: "sombits/evizon-ai-worker"
HF_SPACE_ID = "anon8055/evizon-ai-worker" 

# Initialize Client
try:
    hf_client = Client(HF_SPACE_ID)
    print(f"✅ Connected to AI Worker: {HF_SPACE_ID}")
except Exception as e:
    print(f"⚠️ Warning: Could not connect to Hugging Face. Check your Internet or Space ID. Error: {e}")
    hf_client = None

# ---------- Helper functions ----------
def extract_year_from_text(text: str) -> str:
    if not text: return "Unknown"
    match = re.search(r"(20\d{2})", text)
    return match.group(1) if match else "Unknown"

# ---------- API Endpoints ----------

@app.get("/")
def home():
    # Serve the frontend HTML
    if os.path.exists("frontend/index.html"):
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return {"message": "Frontend file not found, but Backend is active."}

@app.post("/upload")
async def upload_photo(file: UploadFile = File(...)):
    if not hf_client:
        return {"error": "AI Server not connected. Check server logs."}

    # 1. Save locally first
    img_bytes = await file.read()
    photo_id = str(uuid.uuid4())
    filename = file.filename
    temp_path = f"{UPLOAD_DIR}/{photo_id}.jpg"

    with open(temp_path, "wb") as f:
        f.write(img_bytes)

    # 2. Send to Hugging Face for Analysis
    print(f"🚀 Sending {filename} to AI Worker...")
    tags = {}
    
    try:
        # NOTE: If '/predict' fails, check your HF Space "Use via API" button 
        # to see if the name is different (e.g., '/predict_1')
        result = hf_client.predict(
            image_path=handle_file(temp_path),
            api_name="/process_image" 
        )
        tags = result # Result is already a Dict/JSON from our HF app
        print("✅ AI Analysis Complete:", tags)
        
    except Exception as e:
        print(f"❌ AI Error: {e}")
        # IMPORTANT: We don't stop here. We save the file anyway so you don't lose data.
        tags = {"event_text": "Unknown", "is_blurry": False, "hash": ""}

    # 3. Process results
    event_name = tags.get("event_text", "Unknown")
    year = extract_year_from_text(event_name)

    # 4. Organize files
    event_folder = f"{ALBUM_DIR}/{event_name}/{year}"
    os.makedirs(event_folder, exist_ok=True)
    final_path = f"{event_folder}/{filename}"
    
    with open(final_path, "wb") as f:
        f.write(img_bytes)

    # 5. Insert to Supabase
    try:
        supabase.table("event_photos").insert({
            "id": photo_id,
            "photo_url": final_path,
            "event_name": event_name,
            "department": "Unknown",
            "event_date": None,
            "category": "general",
            "cluster_id": -1, 
            "is_blurry": tags.get("is_blurry", False),
            "hash": tags.get("hash", "")
        }).execute()
    except Exception as db_error:
        print(f"⚠️ Database Error: {db_error}")

    # Cleanup temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return {"status": "uploaded", "event": event_name, "year": year}

@app.get("/filter")
def filter_photos(event: str = None, department: str = None, date: str = None):
    query = supabase.table("event_photos").select("*")
    if event: query = query.ilike("event_name", f"%{event}%")
    if department: query = query.eq("department", department)
    if date: query = query.eq("event_date", date)
    return {"results": query.execute().data}

@app.get("/filter-search")
def filter_photos_advanced(event: str = None, department: str = None, date: str = None, category: str = None):
    query = supabase.table("event_photos").select("*")
    if event: query = query.ilike("event_name", f"%{event}%")
    if department: query = query.eq("department", department)
    if date: query = query.eq("event_date", date)
    if category: query = query.eq("category", category)
    
    data = query.execute().data
    return {"count": len(data), "results": data}

@app.get("/search")
def natural_search(q: str = Query(...)):
    q_lower = q.lower()
    dept_match = re.search(r"\b(cse|ece|eee|me)\b", q_lower)
    department = dept_match.group(1).upper() if dept_match else None
    
    query = supabase.table("event_photos").select("*")
    if department: query = query.eq("department", department)
    if not department: query = query.ilike("event_name", f"%{q}%")
         
    results = query.execute().data
    return {"query": q, "results": results}

@app.post("/make-album")
def make_album(event: str):
    rows = supabase.table("event_photos").select("photo_url, event_name").execute().data
    photos = [r["photo_url"] for r in rows if (r["event_name"] or "").lower() == event.lower()]

    zip_path = f"{ALBUM_DIR}/{event}_album.zip"
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for photo in photos:
            if os.path.exists(photo):
                zipf.write(photo, arcname=photo.split("/")[-1])

    return {"album_zip": zip_path, "photo_count": len(photos)}

@app.get("/download-album")
def download_album(event: str = Query(...)):
    zip_path = f"{ALBUM_DIR}/{event}_album.zip"
    if not os.path.exists(zip_path):
        return JSONResponse({"error": "Album ZIP not found"}, status_code=404)
    return FileResponse(zip_path, filename=f"{event}_album.zip", media_type="application/zip")

@app.get("/uploads/{filename}")
def serve_image(filename: str):
    # Try finding it in uploads first
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path)
    
    # If not in uploads, it might be deep inside albums/event/year/filename
    # For the hackathon, we will just try to find it by name walk
    for root, dirs, files in os.walk(ALBUM_DIR):
        if filename in files:
            return FileResponse(os.path.join(root, filename))
            
    return JSONResponse({"error": "Not found"}, 404)

# UI Route for specifically /ui (redirects to home)
@app.get("/ui")
def ui_redirect():
    return home()