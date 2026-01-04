from ultralytics import YOLO
import cv2, imagehash, torch, os, uuid
import easyocr
from sklearn.cluster import DBSCAN
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from fastapi import UploadFile

# Load AI models once (efficient for low-GPU machines too)
yolo_model = YOLO("yolov8n.pt")
ocr_reader = easyocr.Reader(['en'])  # Works best with color images
face_model = InceptionResnetV1(pretrained='vggface2').eval()

def clean_event_name(name: str) -> str:
    if not name:
        return "Unknown"
    name = name.strip()
    replacements = {
        "coni": "Convocation",
        "fest kolkata": "DevFest Kolkata",
        "marathon": "Kolkata Marathon",
    }
    lower = name.lower()
    for k, v in replacements.items():
        if k in lower:
            return v
    return name.title()

async def process_upload(file: UploadFile, save_dir="uploads"):
    # Save image locally
    photo_id = str(uuid.uuid4())
    ext = file.filename.split(".")[-1]
    img_path = os.path.join(save_dir, f"{photo_id}.{ext}")

    os.makedirs(save_dir, exist_ok=True)
    with open(img_path, "wb") as f:
        f.write(await file.read())

    return img_path, photo_id

def analyze_image(img_path):
    tags = {}

    # Load in color to avoid grayscale unpack crash
    img_color = cv2.imread(img_path)
    if img_color is None:
        return {"error": "Image not found"}

    # OCR on color image
    ocr_results = ocr_reader.readtext(img_color)
    raw_text = " ".join([t[1] for t in ocr_results]) if ocr_results else None
    tags["event_text"] = clean_event_name(raw_text)

    # YOLO object detection
    results = yolo_model(img_path)
    detected = [results[0].names[int(box.cls[0])] for box in results[0].boxes]
    tags["objects"] = detected if detected else ["unknown"]

    # Blur detection using grayscale only for Laplacian
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    tags["is_blurry"] = bool(blur_score < 100)

    # Hash generation for duplicates
    img_hash = imagehash.phash(Image.open(img_path))
    tags["hash"] = str(img_hash)

    # Optional cluster placeholder
    tags["cluster_id"] = -1
    tags["is_duplicate"] = False

    return tags

def run_clustering_on_saved_images():
    # Face clustering across uploaded images
    files = os.listdir("uploads")
    embeddings = []
    valid_files = []

    for file in files:
        try:
            img = Image.open(f"uploads/{file}")
            tensor_img = torch.tensor(cv2.imread(f"uploads/{file}")).permute(2,0,1).unsqueeze(0).float()
            emb = face_model(tensor_img).detach().numpy().flatten()
            embeddings.append(emb)
            valid_files.append(file)
        except:
            continue

    if not embeddings:
        return {"error": "No valid images for clustering"}

    clustering = DBSCAN(eps=0.6, min_samples=2).fit(embeddings)
    labels = clustering.labels_

    clusters = {}
    for i, file in enumerate(valid_files):
        clusters[file] = int(labels[i])

    return {"clusters": clusters, "unique_clusters": len(set(labels))}
