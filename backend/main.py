from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import io
import json
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, models
import os

app = FastAPI()

# -------------------------
# CORS (React ↔ FastAPI)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# GLOBAL VARIABLES
# -------------------------
model = None
class_names = []
TAMIL_MAP = {}
DEVICE = torch.device("cpu")

# -------------------------
# LOAD MODEL ON STARTUP
# -------------------------
print("🔄 Loading Model...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "tamil_ocr_model.pth")
mapping_path = os.path.join(BASE_DIR, "class_mapping.json")

try:
    # Load class mapping
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            class_names = json.load(f)
    else:
        class_names = [str(i) for i in range(156)]
        print("⚠️ Warning: class_mapping.json not found. Using default 0-155.")

    # Tamil character map
    TAMIL_MAP = {
        0: "ா", 1: "அ", 2: "ஆ", 3: "இ", 4: "ஈ", 5: "உ", 6: "ஊ", 7: "எ", 8: "ஏ", 9: "ஐ",
        10: "ஒ", 11: "ஓ", 12: "ஔ", 13: "ஃ", 14: "க்", 15: "க", 16: "கி", 17: "கீ", 18: "கு", 19: "கூ",
        20: "ச்", 21: "ச", 22: "சி", 23: "சீ", 24: "சு", 25: "சூ", 26: "ங்", 27: "ங", 28: "ஙி", 29: "ஙீ",
        30: "ஙு", 31: "ஙூ", 32: "ஞ்", 33: "ஞ", 34: "ஞி", 35: "ஞீ", 36: "ஞு", 37: "ஞூ",
        38: "ட்", 39: "ட", 40: "டி", 41: "டீ", 42: "டு", 43: "டூ",
        44: "ண்", 45: "ண", 46: "ணி", 47: "ணீ", 48: "ணு", 49: "ணூ",
        50: "த்", 51: "த", 52: "தி", 53: "தீ", 54: "து", 55: "தூ",
        56: "ந்", 57: "ந", 58: "நி", 59: "நீ", 60: "நு", 61: "நூ",
        62: "ப்", 63: "ப", 64: "பி", 65: "பீ", 66: "பு", 67: "பூ",
        68: "ம்", 69: "ம", 70: "மி", 71: "மீ", 72: "மு", 73: "மூ",
        74: "ய்", 75: "ய", 76: "யி", 77: "யீ", 78: "யு", 79: "யூ",
        80: "ர்", 81: "ர", 82: "ரி", 83: "ரீ", 84: "ரு", 85: "ரூ",
        86: "ல்", 87: "ல", 88: "லி", 89: "லீ", 90: "லு", 91: "லூ",
        92: "ள்", 93: "ள", 94: "ளி", 95: "ளீ", 96: "ளு", 97: "ளூ",
        98: "ற்", 99: "ற", 100: "றி", 101: "றீ", 102: "று", 103: "றூ",
        104: "வ்", 105: "வ", 106: "வி", 107: "வீ", 108: "வு", 109: "வூ",
        110: "ழ்", 111: "ழ", 112: "ழி", 113: "ழீ", 114: "ழு", 115: "ழூ",
        116: "ன்", 117: "ன", 118: "னி", 119: "னீ", 120: "னு",
        121: "ஷி", 122: "ஷீ", 123: "ஷு", 124: "ஷூ",
        125: "க்ஷ", 126: "க்ஷ்", 127: "க்ஷி", 128: "க்ஷீ",
        129: "ஜீ", 130: "ஜூ", 131: "ஹ", 132: "ஹ்",
        133: "ஹி", 134: "ஹீ", 135: "ஹு", 136: "ஹூ",
        137: "ஸ", 138: "ஸ்", 139: "ஸி", 140: "ஸீ",
        141: "ஸு", 142: "ஸூ", 143: "ஷ", 144: "ஷ்",
        145: "னூ", 146: "ஶீ", 147: "க்ஷூ",
        148: "ஜ", 149: "ஜ்", 150: "ஜி", 151: "ஜீ",
        152: "க்ஷு", 153: "ெ", 154: "ே", 155: "ை"
    }

    # Load ResNet18
    if os.path.exists(model_path):
        num_classes = len(class_names)

        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        print(f"✅ Model loaded successfully from: {model_path}")
    else:
        print(f"❌ Model file not found at {model_path}")

except Exception as e:
    print(f"❌ CRITICAL ERROR LOADING MODEL: {e}")

# -------------------------
# IMAGE TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# -------------------------
# PREDICTION ENDPOINT
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        img_cv = np.array(image)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        letters = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 5 and h > 5:
                letters.append((x, y, w, h))

        letters.sort(key=lambda x: x[0])

        result_text = ""

        for (x, y, w, h) in letters:
            roi = gray[max(0, y-5):y+h+5, max(0, x-5):x+w+5]
            pil_roi = Image.fromarray(roi)

            input_tensor = transform(pil_roi).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(input_tensor)
                _, idx = torch.max(output, 1)
                idx = idx.item()

                result_text += TAMIL_MAP.get(idx, "?")

        return {"text": result_text}

    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))
