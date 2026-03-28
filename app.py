from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io, time
from ultralytics import YOLO

app = FastAPI(title="GrocerARy Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load both models at startup
drinks_model = YOLO("best.pt")      # Custom Monster/RedBull model
drinks_model.to("cuda")

coco_model = YOLO("yolov8n.pt")     # COCO model for fruit/all items
coco_model.to("cuda")

# COCO class IDs for fruit/veg
FRUIT_CLASS_IDS = {47, 46, 49, 50, 51}  # apple, banana, orange, broccoli, carrot

# ---------------------------
# Tag rules
# ---------------------------
ORGANIC_LABELS       = {"apple", "banana", "orange", "broccoli", "carrot"}
HEART_HEALTHY_LABELS = {"apple", "banana", "orange", "broccoli", "carrot", "redbull_sugarfree"}
JUNKFOOD_LABELS      = {"monster_blue", "monster_green", "monster_pink", "monster_white", "redbull_sugarfree"}

shopping_list_items: set[str] = set()

def tags_for_label(label: str) -> list[str]:
    l = (label or "").lower()
    tags: list[str] = []
    if l in ORGANIC_LABELS:       tags.append("organic")
    if l in HEART_HEALTHY_LABELS: tags.append("heart_healthy")
    if l in JUNKFOOD_LABELS:      tags.append("junkfood")
    if l in shopping_list_items:  tags.append("shopping_list")
    return tags

# ---------------------------
# Endpoints
# ---------------------------

@app.get("/health")
def health():
    return {"status": "ok", "ts": time.time()}

@app.post("/shopping_list")
async def update_shopping_list(request: Request):
    global shopping_list_items
    data = await request.json()
    shopping_list_items = set(data.get("items", []))
    print(f"Shopping list updated: {shopping_list_items}")
    return {"status": "ok", "items": list(shopping_list_items)}

@app.post("/detect")
async def detect(request: Request):
    data = await request.body()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size

    try:
        conf_threshold = float(request.headers.get("X-Confidence-Threshold", "0.35"))
        conf_threshold = max(0.05, min(0.95, conf_threshold))
    except (ValueError, TypeError):
        conf_threshold = 0.35

    # Read detection mode from header — sent by Android app
    detection_mode = request.headers.get("X-Detection-Mode", "drinks").lower()

    t0 = time.time()

    if detection_mode == "drinks":
        # Use custom Monster/RedBull model — no class filtering needed
        results = drinks_model.predict(img, verbose=False, imgsz=640, conf=conf_threshold)
        r = results[0]
        detections = []
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf  = float(b.conf[0])
                label = r.names.get(int(b.cls[0]), "unknown")
                detections.append({
                    "label": label,
                    "conf":  conf,
                    "bbox":  [x1/w, y1/h, (x2-x1)/w, (y2-y1)/h],
                    "tags":  tags_for_label(label)
                })

    elif detection_mode == "fruit":
        # Use COCO model filtered to fruit/veg only
        results = coco_model.predict(img, verbose=False, imgsz=640, conf=conf_threshold)
        r = results[0]
        detections = []
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                cls_id = int(b.cls[0])
                if cls_id not in FRUIT_CLASS_IDS:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf  = float(b.conf[0])
                label = r.names.get(cls_id, "unknown")
                detections.append({
                    "label": label,
                    "conf":  conf,
                    "bbox":  [x1/w, y1/h, (x2-x1)/w, (y2-y1)/h],
                    "tags":  tags_for_label(label)
                })

    else:
        # "all" mode — full COCO 80 classes, no filter
        results = coco_model.predict(img, verbose=False, imgsz=640, conf=conf_threshold)
        r = results[0]
        detections = []
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf  = float(b.conf[0])
                label = r.names.get(int(b.cls[0]), "unknown")
                detections.append({
                    "label": label,
                    "conf":  conf,
                    "bbox":  [x1/w, y1/h, (x2-x1)/w, (y2-y1)/h],
                    "tags":  tags_for_label(label)
                })

    infer_ms = int((time.time() - t0) * 1000)

    print(f"YOLO detections: {len(detections)} | mode: {detection_mode} | infer: {infer_ms}ms | conf: {conf_threshold}")
    return {
        "frame_id":   int(time.time() * 1000),
        "infer_ms":   infer_ms,
        "image_size": {"width": w, "height": h},
        "detections": detections
    }