from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
import uvicorn
import os

from model.model import load_model, predict_crack

app = FastAPI()

# Allow frontend access from any device (for camera)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("app/static/index.html", "r") as f:
        return HTMLResponse(f.read())

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    result = predict_crack(model, image)
    return {"prediction": result}
