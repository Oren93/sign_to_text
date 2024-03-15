from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from predict import make_prediction

UPLOAD_DIR = Path("../uploads")

app = FastAPI()

templates = Jinja2Templates(directory = "../frontend")
app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")

@app.post('/uploadfile/')
async def create_upload_file(file_upload: UploadFile, request: Request):
    data = await file_upload.read()
    path = UPLOAD_DIR / file_upload.filename
    with open(path, "wb") as f:
        f.write(data)
    #predicted_word, confidence = make_prediction(path)
    predicted_word = "short"
    confidence = 0

    VALUES = [{"filename": file_upload.filename, "predicted_word": predicted_word, "confidence_level": confidence}]
    return templates.TemplateResponse("index.html", {"request": request, "values": VALUES})
