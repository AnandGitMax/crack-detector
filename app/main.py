# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import HTMLResponse
# from io import BytesIO
# from PIL import Image
# import torch
# import torchvision.transforms as transforms
# import uvicorn
# import os

# #from model.model import load_model, predict_crack
# from app.model.model import load_model, predict_crack


# app = FastAPI()

# # Allow frontend access from any device (for camera)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# model = load_model()

# @app.get("/", response_class=HTMLResponse)
# async def home():
#     with open("app/static/index.html", "r") as f:
#         return HTMLResponse(f.read())

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     image = Image.open(BytesIO(await file.read())).convert("RGB")
#     result = predict_crack(model, image)
#     return {"prediction": result}

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
# Removed unused import: import uvicorn
import os

# Note: Keeping the import as app.model.model based on your last success
from app.model.model import load_model, predict_crack 


app = FastAPI()

# Allow frontend access from any device
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once when the application starts
try:
    model = load_model()
    print("INFO: Crack detection model loaded successfully.")
except Exception as e:
    # Log a critical error if the model fails to load
    print(f"CRITICAL ERROR: Failed to load crack detection model: {e}")
    model = None # Set model to None so requests will fail safely


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serves the index.html file for the frontend."""
    # Ensure the path is correct relative to the execution location
    try:
        with open("app/static/index.html", "r") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>Error: index.html not found at app/static/index.html</h1>", status_code=500)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Receives image, runs prediction, and returns result with error handling."""
    if model is None:
        return {"prediction": "ERROR: Model failed to load at startup."}
    
    try:
        # 1. Read the image data from the uploaded file
        image_bytes = await file.read()
        
        # 2. Open the image using PIL
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # 3. Run the prediction (This is the critical section where errors usually occur)
        result = predict_crack(model, image)
        
        # 4. Return the result dictionary expected by the JavaScript frontend
        return {"prediction": result}
        
    except Exception as e:
        # Log the full error to the Uvicorn console for debugging
        print(f"ERROR: Prediction failed for file {file.filename}: {e}")
        # Return an error message to the frontend
        return {"prediction": f"ERROR: Processing failed. Check server logs."}