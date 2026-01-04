from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from ultralytics import YOLO
import easyocr, cv2, os, uuid, zipfile, imagehash, torch
from PIL import Image
from sklearn.cluster import DBSCAN
from facenet_pytorch import InceptionResnetV1
from supabase_client import supabase  # already configured
from search_parser import parse_query
from PIL import Image
import numpy as np
import re



app = FastAPI()

UPLOAD_DIR = "uploads"
ALBUM_DIR = "albums"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ALBUM_DIR, exist_ok=True)

# Load AI models once
yolo_model = YOLO("yolov8n.pt")
ocr_reader = easyocr.Reader(['en'])
face_model = InceptionResnetV1(pretrained='vggface2').eval()

# ---------- Helper functions ----------
def clean_event_name(name: str) -> str:
    if not name:
        return "Unknown"
    name = name.strip()
    replacements = {
        "coni": "Convocation",
        "fest kolkata": "DevFest Kolkata",
        "marathon": "Kolkata Marathon",
        "devfest": "DevFest",
        "gdg kolkata": "Google Dev Kolkata",
    }
    lower = name.lower()
    for k, v in replacements.items():
        if k in lower:
            return v
    return name.title()

def extract_year_from_text(text: str) -> str:
    import re
    if not text:
        return "Unknown"
    match = re.search(r"(20\d{2})", text)
    return match.group(1) if match else "Unknown"

def analyze_image(img_path):
    tags = {}

    img_color = cv2.imread(img_path)
    if img_color is None:
        return {"error": "Image not found"}

    # OCR FIX → send image array, not file path
    ocr_results = ocr_reader.readtext(img_color)
    raw_text = " ".join([t[1] for t in ocr_results]) if ocr_results else None
    tags["event_text"] = clean_event_name(raw_text)

    # YOLO object detection
    results = yolo_model(img_path)
    detected = [results[0].names[int(box.cls[0])] for box in results[0].boxes]
    tags["objects"] = detected if detected else ["unknown"]

    # Blur detection
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    tags["is_blurry"] = bool(blur_score < 100)

    # Hash generation
    tags["hash"] = str(imagehash.phash(Image.open(img_path)))

    return tags


# ---------- API Endpoints ----------

@app.get("/")
def home():
    return {"message": "Evizon backend active, Admin."}

@app.post("/upload")
async def upload_photo(file: UploadFile = File(...)):
    img_bytes = await file.read()
    photo_id = str(uuid.uuid4())
    filename = file.filename
    temp_path = f"{UPLOAD_DIR}/{photo_id}.jpg"

    with open(temp_path, "wb") as f:
        f.write(img_bytes)

    tags = analyze_image(temp_path)
    event_name = tags["event_text"]
    year = extract_year_from_text(event_name)

    event_folder = f"{ALBUM_DIR}/{event_name}/{year}"
    os.makedirs(event_folder, exist_ok=True)

    final_path = f"{event_folder}/{filename}"
    with open(final_path, "wb") as f:
        f.write(img_bytes)

    # Insert metadata into Supabase
    supabase.table("event_photos").insert({
        "id": photo_id,
        "photo_url": final_path,
        "event_name": event_name,
        "department": "Unknown",
        "event_date": None,
        "category": "general",
        "cluster_id": -1,
        "is_blurry": tags["is_blurry"],
        "hash": tags["hash"]
    }).execute()

    # Remove temp file
    os.remove(temp_path)

    return {"status": "uploaded", "event": event_name, "year": year, "saved_in": final_path}


@app.get("/filter")
def filter_photos(
    event: str = None,
    department: str = None,
    date: str = None
):
    query = supabase.table("event_photos").select("id, photo_url, event_name, department, event_date, category")

    if event:
        query = query.ilike("event_name", f"%{event}%")
    if department:
        query = query.eq("department", department)
    if date:
        query = query.eq("event_date", date)

    results = query.execute().data
    return {"count": len(results), "results": results}

@app.get("/filter-search")
def filter_photos_advanced(
    event: str = None,
    department: str = None,
    date: str = None,
    category: str = None
):
    query = supabase.table("event_photos").select(
        "id, photo_url, event_name, department, event_date, category, hash, is_blurry, cluster_id"
    )

    if event:
        query = query.ilike("event_name", f"%{event}%")
    if department:
        query = query.eq("department", department)
    if date:
        query = query.eq("event_date", date)
    if category:
        query = query.eq("category", category)

    results = query.execute().data

    return {"count": len(results), "results": results}

@app.get("/search")
def natural_search(q: str = Query(..., description="Search query")):

    # Decode intent from query
    q_lower = q.lower()

    # Extract department if present
    dept_match = re.search(r"\b(cse|ece|eee|me)\b", q_lower)
    department = dept_match.group(1).upper() if dept_match else None

    # Extract year if present
    year_match = re.search(r"\b20\d{2}\b", q_lower)
    year = year_match.group(0) if year_match else None

    # Clean event name from text using your function
    event = clean_event_name(q)

    # Build DB query
    query = supabase.table("event_photos").select(
        "id, photo_url, event_name, department, event_date, category, hash, is_blurry, cluster_id"
    )

    if department:
        query = query.eq("department", department)
    if year and year != "Unknown":
        query = query.ilike("event_name", f"%{year}%")
    if event and event != "Unknown":
        query = query.ilike("event_name", f"%{event}%")

    results = query.execute().data

    return {
        "query": q,
        "detected_event": event,
        "detected_department": department,
        "count": len(results),
        "results": results
    }

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

    return FileResponse(
        zip_path,
        filename=f"{event}_album.zip",
        media_type="application/zip"
    )

@app.get("/tags")
def get_tags(photo_id: str = Query(...)):
    data = supabase.table("event_photos").select("hash, cluster_id, is_blurry, category, event_name").eq("id", photo_id).execute().data
    return {"tags": data}

@app.post("/cluster")
def cluster_faces(eps: float = 0.6, min_samples: int = 2):
    # Fetch all stored photos from DB
    rows = supabase.table("event_photos").select("id, photo_url").execute().data
    if not rows:
        return {"error": "No photos in database"}

    embeddings = []
    ids = []

    # Generate embeddings
    for r in rows:
        try:
            img = Image.open(r["photo_url"]).convert("RGB")
            img_tensor = torch.unsqueeze(torch.tensor(np.array(img)).permute(2,0,1).float(), 0)
            emb = face_model(img_tensor).detach().numpy().flatten()
            embeddings.append(emb)
            ids.append(r["id"])
        except Exception as e:
            print("Embedding failed for:", r["photo_url"], "Error:", e)
            continue

    if not embeddings:
        return {"error": "Embedding generation failed for all images"}

    # Run clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    labels = clustering.labels_

    # Update cluster_id in DB for each photo
    for i, pid in enumerate(ids):
        supabase.table("event_photos").update({"cluster_id": int(labels[i])}).eq("id", pid).execute()

    return {
        "clusters": list(map(int, labels)),
        "unique": len(set(labels)) - (1 if -1 in labels else 0)
    }

@app.get("/uploads/{filename}")
def serve_image(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "Image not found"}, status_code=404)
    return FileResponse(path, media_type="image/jpeg")


# Optional: Serve UI if needed
@app.get("/ui")
def serve_ui():
    if os.path.exists("frontend/index.html"):
        with open("frontend/index.html", "r") as f:
            return HTMLResponse(f.read())
    return {"message": "UI file missing"}
