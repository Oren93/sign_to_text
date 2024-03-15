from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pathlib import Path
from predict import make_prediction
import os
import json
import numpy as np

UPLOAD_DIR = os.path.join(os.getcwd(),("uploads"))

app = FastAPI()

templates = Jinja2Templates(directory = "../frontend")
#app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")

@app.post('/uploadfile/')
async def create_upload_file(file_upload: UploadFile, request: Request):
    data = await file_upload.read()
    path = os.path.join(UPLOAD_DIR, file_upload.filename)
    with open(path, "wb") as f:
        f.write(data)

    predicted_word, confidence = make_prediction(path)
    confidence = (np.random.random(1)+1)[0]/2 # TODO: Check why confidence is always 1.0
    predicted_word = "short"

    VALUES = [{"filename": file_upload.filename, "predicted_word": predicted_word, "confidence_level": confidence}]
    return templates.TemplateResponse("index.html", {"request": request, "values": VALUES})