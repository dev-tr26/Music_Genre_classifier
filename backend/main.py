from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
import os
import shutil
import asyncio
import uuid

from preprocessing import preprocess_long_audio
from loader import load_model
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.concurrency import run_in_threadpool

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


model = load_model()

classes = ["blues", "classical", "country", "disco","hiphop", "jazz", "metal", "pop", "reggae", "rock"]


uploaded_files: dict[str, str] = {}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.get("/result", response_class=HTMLResponse)
def result_page(request: Request):
    return templates.TemplateResponse("result.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[-1]
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        audio_chunks = await run_in_threadpool(preprocess_long_audio,file_path)
        print("Chunks shape:", audio_chunks.shape)
        with torch.no_grad():
            outputs = await run_in_threadpool(model,audio_chunks)
            print("Model outputs:")
            print(outputs)
            probabilities = F.softmax(outputs, dim=1)
            print(probabilities)
            avg_probabilities = probabilities.mean(dim=0)
            pred = torch.argmax(avg_probabilities).item()

            confidence = {
                classes[i]: round(float(avg_probabilities[i]) * 100, 2)
                for i in range(len(classes))
            }

        # Sorted confidence descending
        confidence = dict(sorted(confidence.items(), key=lambda x: x[1], reverse=True))

        return JSONResponse({
            "predicted_genre": classes[pred],
            "confidence": confidence,
            "file_id": unique_name
        })

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.delete("/cleanup/{file_id}")
async def cleanup(file_id: str):
    safe_name = os.path.basename(file_id)
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)
    if os.path.exists(file_path):
        os.remove(file_path)
    return {"status": "cleaned"}